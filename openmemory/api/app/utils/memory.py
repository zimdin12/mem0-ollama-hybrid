"""
Memory client utilities for OpenMemory.

This module provides functionality to initialize and manage the Mem0 memory client
with automatic configuration management and Docker environment support.

Docker Ollama Configuration:
When running inside a Docker container and using Ollama as the LLM or embedder provider,
the system automatically detects the Docker environment and adjusts localhost URLs
to properly reach the host machine where Ollama is running.

Supported Docker host resolution (in order of preference):
1. OLLAMA_HOST environment variable (if set)
2. host.docker.internal (Docker Desktop for Mac/Windows)
3. Docker bridge gateway IP (typically 172.17.0.1 on Linux)
4. Fallback to 172.17.0.1

Example configuration that will be automatically adjusted:
{
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "ollama_base_url": "http://localhost:11434"  # Auto-adjusted in Docker
        }
    }
}
"""

import hashlib
import json
import os
import socket

from app.database import SessionLocal
from app.models import Config as ConfigModel

from mem0 import Memory

_memory_client = None
_config_hash = None

# ========================================================================
# ESSENTIAL FIX FOR OLLAMA + MEM0 COMPATIBILITY
# ========================================================================

def _apply_essential_ollama_fix():
    """Essential fix: ensure OllamaLLM returns strings, not dicts"""
    try:
        from mem0.llms.ollama import OllamaLLM
        from functools import wraps
        import json
        
        if not hasattr(OllamaLLM, 'generate_response'):
            return False
        
        original_generate = OllamaLLM.generate_response
        
        def _parse_content_to_tool_calls(content, tools):
            """Parse qwen3:4b content into mem0-expected tool_calls format"""
            import json
            import re
            
            tool_calls = []
            
            if not tools or not content:
                return tool_calls
            
            try:
                # Try to parse as JSON first
                if content.strip().startswith('{'):
                    parsed = json.loads(content)
                    
                    # Look for entities in the parsed JSON
                    if 'entities' in parsed:
                        entities = parsed['entities']
                        
                        # Convert to proper format for extract_entities tool
                        for tool in tools:
                            if tool.get('function', {}).get('name') == 'extract_entities':
                                tool_calls.append({
                                    'name': 'extract_entities',
                                    'arguments': {
                                        'entities': [
                                            {
                                                'entity': ent.get('entity', ent) if isinstance(ent, dict) else ent,
                                                'entity_type': ent.get('type', 'MISC') if isinstance(ent, dict) else 'MISC'
                                            }
                                            for ent in entities
                                        ]
                                    }
                                })
                                break
                            elif tool.get('function', {}).get('name') == 'establish_relationships':
                                # Handle relationship extraction
                                if 'relationships' in parsed:
                                    relationships = parsed.get('relationships', [])
                                    tool_calls.append({
                                        'name': 'establish_relationships',
                                        'arguments': {
                                            'entities': [
                                                {
                                                    'source': rel.get('subject', rel.get('source', 'unknown')),
                                                    'relationship': rel.get('predicate', rel.get('relationship', 'related_to')),
                                                    'destination': rel.get('object', rel.get('target', rel.get('destination', 'unknown')))
                                                }
                                                for rel in relationships
                                            ]
                                        }
                                    })
                else:
                    # Try to extract entities from text format
                    lines = content.split('\n')
                    entities = []
                    
                    for line in lines:
                        # Look for patterns like "- Sarah Johnson: Person"
                        match = re.match(r'-\s*([^:]+):\s*(.+)', line.strip())
                        if match:
                            entity = match.group(1).strip()
                            entity_type = match.group(2).strip()
                            entities.append({
                                'entity': entity,
                                'entity_type': entity_type
                            })
                    
                    if entities:
                        for tool in tools:
                            if tool.get('function', {}).get('name') == 'extract_entities':
                                tool_calls.append({
                                    'name': 'extract_entities',
                                    'arguments': {'entities': entities}
                                })
                                break
                                
            except Exception as e:
                print(f"⚠️  Tool call parsing error: {e}")
            
            return tool_calls
        
        @wraps(original_generate)
        def string_response_generate(self, messages, response_format=None, tools=None, tool_choice='auto', **kwargs):
            result = original_generate(self, messages=messages, response_format=response_format, tools=tools, tool_choice=tool_choice, **kwargs)
            
            # The key insight: if tools are provided, this is for entity extraction (return dict)
            # If no tools, this is for fact extraction (return string)
            
            if tools is not None:
                # Entity extraction pipeline - need to create proper tool_calls format
                if isinstance(result, dict):
                    if 'tool_calls' not in result or not result['tool_calls']:
                        # Parse the content and create tool_calls format
                        content = result.get('content', '')
                        if content:
                            tool_calls = _parse_content_to_tool_calls(content, tools)
                            result['tool_calls'] = tool_calls
                    return result
                else:
                    # Parse string content and create tool_calls
                    tool_calls = _parse_content_to_tool_calls(str(result), tools)
                    return {
                        'content': str(result),
                        'tool_calls': tool_calls
                    }
            else:
                # Fact extraction pipeline - return strings (fixes the strip error)
                if isinstance(result, dict) and 'content' in result:
                    return result['content']  # Return just the content string
                elif isinstance(result, dict):
                    return json.dumps(result)  # Convert dict to JSON string
                else:
                    return str(result)  # Ensure it's a string
        
        OllamaLLM.generate_response = string_response_generate
        print("✅ Applied essential Ollama fix")
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply essential fix: {e}")
        return False

# Apply the essential fix
_apply_essential_ollama_fix()


def _get_config_hash(config_dict):
    """Generate a hash of the config to detect changes."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def _get_docker_host_url():
    """
    Determine the appropriate host URL to reach host machine from inside Docker container.
    Returns the best available option for reaching the host from inside a container.
    """
    # Check for custom environment variable first
    custom_host = os.environ.get('OLLAMA_HOST')
    if custom_host:
        print(f"Using custom Ollama host from OLLAMA_HOST: {custom_host}")
        return custom_host.replace('http://', '').replace('https://', '').split(':')[0]
    
    # Check if we're running inside Docker
    if not os.path.exists('/.dockerenv'):
        # Not in Docker, return localhost as-is
        return "localhost"
    
    print("Detected Docker environment, adjusting host URL for Ollama...")
    
    # Try different host resolution strategies
    host_candidates = []
    
    # 1. host.docker.internal (works on Docker Desktop for Mac/Windows)
    try:
        socket.gethostbyname('host.docker.internal')
        host_candidates.append('host.docker.internal')
        print("Found host.docker.internal")
    except socket.gaierror:
        pass
    
    # 2. Docker bridge gateway (typically 172.17.0.1 on Linux)
    try:
        with open('/proc/net/route', 'r') as f:
            for line in f:
                fields = line.strip().split()
                if fields[1] == '00000000':  # Default route
                    gateway_hex = fields[2]
                    gateway_ip = socket.inet_ntoa(bytes.fromhex(gateway_hex)[::-1])
                    host_candidates.append(gateway_ip)
                    print(f"Found Docker gateway: {gateway_ip}")
                    break
    except (FileNotFoundError, IndexError, ValueError):
        pass
    
    # 3. Fallback to common Docker bridge IP
    if not host_candidates:
        host_candidates.append('172.17.0.1')
        print("Using fallback Docker bridge IP: 172.17.0.1")
    
    # Return the first available candidate
    return host_candidates[0]


def _fix_ollama_urls(config_section):
    """
    Fix Ollama URLs for Docker environment.
    Replaces localhost URLs with appropriate Docker host URLs.
    Sets default ollama_base_url if not provided.
    """
    if not config_section or "config" not in config_section:
        return config_section
    
    ollama_config = config_section["config"]
    
    # Set default ollama_base_url if not provided
    if "ollama_base_url" not in ollama_config:
        ollama_config["ollama_base_url"] = "http://host.docker.internal:11434"
    else:
        # Check for ollama_base_url and fix if it's localhost
        url = ollama_config["ollama_base_url"]
        if "localhost" in url or "127.0.0.1" in url:
            docker_host = _get_docker_host_url()
            if docker_host != "localhost":
                new_url = url.replace("localhost", docker_host).replace("127.0.0.1", docker_host)
                ollama_config["ollama_base_url"] = new_url
                print(f"Adjusted Ollama URL from {url} to {new_url}")
    
    return config_section


def reset_memory_client():
    """Reset the global memory client to force reinitialization with new config."""
    global _memory_client, _config_hash
    _memory_client = None
    _config_hash = None


def get_default_memory_config():
    """Get default memory client configuration with sensible defaults."""
    # Detect vector store based on environment variables
    vector_store_config = {
        "collection_name": "openmemory",
        "host": "mem0_store",
    }
    
    # Check for different vector store configurations based on environment variables
    if os.environ.get('CHROMA_HOST') and os.environ.get('CHROMA_PORT'):
        vector_store_provider = "chroma"
        vector_store_config.update({
            "host": os.environ.get('CHROMA_HOST'),
            "port": int(os.environ.get('CHROMA_PORT'))
        })
    elif os.environ.get('QDRANT_HOST') and os.environ.get('QDRANT_PORT'):
        vector_store_provider = "qdrant"
        vector_store_config.update({
            "host": os.environ.get('QDRANT_HOST'),
            "port": int(os.environ.get('QDRANT_PORT'))
        })
    elif os.environ.get('WEAVIATE_CLUSTER_URL') or (os.environ.get('WEAVIATE_HOST') and os.environ.get('WEAVIATE_PORT')):
        vector_store_provider = "weaviate"
        # Prefer an explicit cluster URL if provided; otherwise build from host/port
        cluster_url = os.environ.get('WEAVIATE_CLUSTER_URL')
        if not cluster_url:
            weaviate_host = os.environ.get('WEAVIATE_HOST')
            weaviate_port = int(os.environ.get('WEAVIATE_PORT'))
            cluster_url = f"http://{weaviate_host}:{weaviate_port}"
        vector_store_config = {
            "collection_name": "openmemory",
            "cluster_url": cluster_url
        }
    elif os.environ.get('REDIS_URL'):
        vector_store_provider = "redis"
        vector_store_config = {
            "collection_name": "openmemory",
            "redis_url": os.environ.get('REDIS_URL')
        }
    elif os.environ.get('PG_HOST') and os.environ.get('PG_PORT'):
        vector_store_provider = "pgvector"
        vector_store_config.update({
            "host": os.environ.get('PG_HOST'),
            "port": int(os.environ.get('PG_PORT')),
            "dbname": os.environ.get('PG_DB', 'mem0'),
            "user": os.environ.get('PG_USER', 'mem0'),
            "password": os.environ.get('PG_PASSWORD', 'mem0')
        })
    elif os.environ.get('MILVUS_HOST') and os.environ.get('MILVUS_PORT'):
        vector_store_provider = "milvus"
        # Construct the full URL as expected by MilvusDBConfig
        milvus_host = os.environ.get('MILVUS_HOST')
        milvus_port = int(os.environ.get('MILVUS_PORT'))
        milvus_url = f"http://{milvus_host}:{milvus_port}"
        
        vector_store_config = {
            "collection_name": "openmemory",
            "url": milvus_url,
            "token": os.environ.get('MILVUS_TOKEN', ''),  # Always include, empty string for local setup
            "db_name": os.environ.get('MILVUS_DB_NAME', ''),
            "embedding_model_dims": 1536,
            "metric_type": "COSINE"  # Using COSINE for better semantic similarity
        }
    elif os.environ.get('ELASTICSEARCH_HOST') and os.environ.get('ELASTICSEARCH_PORT'):
        vector_store_provider = "elasticsearch"
        # Construct the full URL with scheme since Elasticsearch client expects it
        elasticsearch_host = os.environ.get('ELASTICSEARCH_HOST')
        elasticsearch_port = int(os.environ.get('ELASTICSEARCH_PORT'))
        # Use http:// scheme since we're not using SSL
        full_host = f"http://{elasticsearch_host}"
        
        vector_store_config.update({
            "host": full_host,
            "port": elasticsearch_port,
            "user": os.environ.get('ELASTICSEARCH_USER', 'elastic'),
            "password": os.environ.get('ELASTICSEARCH_PASSWORD', 'changeme'),
            "verify_certs": False,
            "use_ssl": False,
            "embedding_model_dims": 1536
        })
    elif os.environ.get('OPENSEARCH_HOST') and os.environ.get('OPENSEARCH_PORT'):
        vector_store_provider = "opensearch"
        vector_store_config.update({
            "host": os.environ.get('OPENSEARCH_HOST'),
            "port": int(os.environ.get('OPENSEARCH_PORT'))
        })
    elif os.environ.get('FAISS_PATH'):
        vector_store_provider = "faiss"
        vector_store_config = {
            "collection_name": "openmemory",
            "path": os.environ.get('FAISS_PATH'),
            "embedding_model_dims": 1536,
            "distance_strategy": "cosine"
        }
    else:
        # Default fallback to Qdrant
        vector_store_provider = "qdrant"
        vector_store_config.update({
            "port": 6333,
        })
    
    print(f"Auto-detected vector store: {vector_store_provider} with config: {vector_store_config}")
    
    return {
        "vector_store": {
            "provider": vector_store_provider,
            "config": vector_store_config
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000,
                "api_key": "env:OPENAI_API_KEY"
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": "env:OPENAI_API_KEY"
            }
        },
        "version": "v1.1"
    }


def _parse_environment_variables(config_dict):
    """
    Parse environment variables in config values.
    Converts 'env:VARIABLE_NAME' to actual environment variable values.
    """
    if isinstance(config_dict, dict):
        parsed_config = {}
        for key, value in config_dict.items():
            if isinstance(value, str) and value.startswith("env:"):
                env_var = value.split(":", 1)[1]
                env_value = os.environ.get(env_var)
                if env_value:
                    parsed_config[key] = env_value
                    print(f"Loaded {env_var} from environment for {key}")
                else:
                    print(f"Warning: Environment variable {env_var} not found, keeping original value")
                    parsed_config[key] = value
            elif isinstance(value, dict):
                parsed_config[key] = _parse_environment_variables(value)
            else:
                parsed_config[key] = value
        return parsed_config
    return config_dict


def get_memory_client(custom_instructions: str = None):
    """
    Get or initialize the Mem0 client.

    Args:
        custom_instructions: Optional instructions for the memory project.

    Returns:
        Initialized Mem0 client instance or None if initialization fails.

    Raises:
        Exception: If required API keys are not set or critical configuration is missing.
    """
    global _memory_client, _config_hash

    try:
        # Start with default configuration
        config = get_default_memory_config()
        
        # Variable to track custom instructions
        db_custom_instructions = None
        
        # TRY TO LOAD FROM config.json FIRST
        config_loaded_from = "defaults"
        try:
            with open('config.json', 'r') as f:
                json_config = json.load(f)
                print("✓ Loaded configuration from config.json")
                
                # Extract custom instructions from openmemory settings
                if "openmemory" in json_config and "custom_instructions" in json_config["openmemory"]:
                    db_custom_instructions = json_config["openmemory"]["custom_instructions"]
                
                # Override defaults with configurations from config.json
                if "mem0" in json_config:
                    mem0_config = json_config["mem0"]
                    
                    # Update LLM configuration if available
                    if "llm" in mem0_config and mem0_config["llm"] is not None:
                        config["llm"] = mem0_config["llm"]
                        if config["llm"].get("provider") == "ollama":
                            config["llm"] = _fix_ollama_urls(config["llm"])
                    
                    # Update Embedder configuration if available
                    if "embedder" in mem0_config and mem0_config["embedder"] is not None:
                        config["embedder"] = mem0_config["embedder"]
                        if config["embedder"].get("provider") == "ollama":
                            config["embedder"] = _fix_ollama_urls(config["embedder"])

                    # Update Vector Store configuration if available
                    if "vector_store" in mem0_config and mem0_config["vector_store"] is not None:
                        config["vector_store"] = mem0_config["vector_store"]
                    
                    # Load graph_store from config.json
                    if "graph_store" in mem0_config and mem0_config["graph_store"] is not None:
                        config["graph_store"] = mem0_config["graph_store"]
                        print(f"✓ Loaded graph_store config: {config['graph_store']}")
                
                config_loaded_from = "config.json"
        except FileNotFoundError:
            print("config.json not found, trying database...")
        except Exception as e:
            print(f"Warning: Error loading config.json: {e}")
        
        # Load configuration from database (this will override config.json if present)
        try:
            db = SessionLocal()
            db_config = db.query(ConfigModel).filter(ConfigModel.key == "main").first()
            
            if db_config:
                json_config = db_config.value
                print("✓ Loading configuration from database (overriding config.json)")
                
                # Extract custom instructions from openmemory settings
                if "openmemory" in json_config and "custom_instructions" in json_config["openmemory"]:
                    db_custom_instructions = json_config["openmemory"]["custom_instructions"]
                
                # Override with configurations from the database
                if "mem0" in json_config:
                    mem0_config = json_config["mem0"]
                    
                    # Update LLM configuration if available
                    if "llm" in mem0_config and mem0_config["llm"] is not None:
                        config["llm"] = mem0_config["llm"]
                        
                        # Fix Ollama URLs for Docker if needed
                        if config["llm"].get("provider") == "ollama":
                            config["llm"] = _fix_ollama_urls(config["llm"])
                    
                    # Update Embedder configuration if available
                    if "embedder" in mem0_config and mem0_config["embedder"] is not None:
                        config["embedder"] = mem0_config["embedder"]
                        
                        # Fix Ollama URLs for Docker if needed
                        if config["embedder"].get("provider") == "ollama":
                            config["embedder"] = _fix_ollama_urls(config["embedder"])

                    if "vector_store" in mem0_config and mem0_config["vector_store"] is not None:
                        config["vector_store"] = mem0_config["vector_store"]
                    
                    # Load graph_store from config.json/database
                    if "graph_store" in mem0_config and mem0_config["graph_store"] is not None:
                        config["graph_store"] = mem0_config["graph_store"]
                        print(f"✓ Loaded graph_store config: {config['graph_store']}")
                config_loaded_from = "database"
            else:
                print(f"No configuration found in database, using {config_loaded_from}")
                    
            db.close()
                            
        except Exception as e:
            print(f"Warning: Error loading configuration from database: {e}")
            print(f"Using configuration from {config_loaded_from}")
            # Continue with default configuration if database config can't be loaded

        # Use custom_instructions parameter first, then fall back to database value
        instructions_to_use = custom_instructions or db_custom_instructions
        if instructions_to_use:
            config["custom_fact_extraction_prompt"] = instructions_to_use

        # ALWAYS parse environment variables in the final config
        # This ensures that even default config values like "env:OPENAI_API_KEY" get parsed
        print("Parsing environment variables in final config...")
        config = _parse_environment_variables(config)
        
        # Ensure graph_store has proper LLM configuration for entity extraction
        if "graph_store" in config:
            if "llm" not in config["graph_store"] or config["graph_store"]["llm"] is None:
                config["graph_store"]["llm"] = config["llm"]
                print(f"✓ Added parsed LLM config to graph_store for entity extraction")

        # Check if config has changed by comparing hashes
        current_config_hash = _get_config_hash(config)
        
        # Only reinitialize if config changed or client doesn't exist
        if _memory_client is None or _config_hash != current_config_hash:
            print(f"Initializing memory client with config hash: {current_config_hash}")
            try:
                _memory_client = Memory.from_config(config_dict=config)
                _config_hash = current_config_hash
                print("Memory client initialized successfully")
            except Exception as init_error:
                print(f"Warning: Failed to initialize memory client: {init_error}")
                print("Server will continue running with limited memory functionality")
                _memory_client = None
                _config_hash = None
                return None
        
        return _memory_client
        
    except Exception as e:
        print(f"Warning: Exception occurred while initializing memory client: {e}")
        print("Server will continue running with limited memory functionality")
        return None


def get_default_user_id():
    return "default_user"
