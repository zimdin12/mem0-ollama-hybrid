import json
import os
from typing import Any, Dict, Optional

from app.database import get_db
from app.models import Config as ConfigModel
from app.utils.memory import reset_memory_client
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1/config", tags=["config"])

class LLMConfig(BaseModel):
    model: str = Field(..., description="LLM model name")
    temperature: float = Field(..., description="Temperature setting for the model")
    max_tokens: int = Field(..., description="Maximum tokens to generate")
    api_key: Optional[str] = Field(None, description="API key or 'env:API_KEY' to use environment variable")
    ollama_base_url: Optional[str] = Field(None, description="Base URL for Ollama server (e.g., http://host.docker.internal:11434)")

class LLMProvider(BaseModel):
    provider: str = Field(..., description="LLM provider name")
    config: LLMConfig

class EmbedderConfig(BaseModel):
    model: str = Field(..., description="Embedder model name")
    api_key: Optional[str] = Field(None, description="API key or 'env:API_KEY' to use environment variable")
    ollama_base_url: Optional[str] = Field(None, description="Base URL for Ollama server (e.g., http://host.docker.internal:11434)")

class EmbedderProvider(BaseModel):
    provider: str = Field(..., description="Embedder provider name")
    config: EmbedderConfig

class VectorStoreProvider(BaseModel):
    provider: str = Field(..., description="Vector store provider name")
    # Below config can vary widely based on the vector store used. Refer https://docs.mem0.ai/components/vectordbs/config
    config: Dict[str, Any] = Field(..., description="Vector store-specific configuration")

class OpenMemoryConfig(BaseModel):
    custom_instructions: Optional[str] = Field(None, description="Custom instructions for memory management and fact extraction")

class Mem0Config(BaseModel):
    llm: Optional[LLMProvider] = None
    embedder: Optional[EmbedderProvider] = None
    vector_store: Optional[VectorStoreProvider] = None

class ConfigSchema(BaseModel):
    openmemory: Optional[OpenMemoryConfig] = None
    mem0: Optional[Mem0Config] = None

def get_default_configuration():
    """Build defaults from config.json and environment variables."""
    config = {
        "openmemory": {"custom_instructions": None},
        "mem0": {"llm": None, "embedder": None, "vector_store": None}
    }

    # Try loading from config.json first
    try:
        with open('config.json', 'r') as f:
            json_config = json.load(f)
        if "mem0" in json_config:
            if "llm" in json_config["mem0"]:
                config["mem0"]["llm"] = json_config["mem0"]["llm"]
            if "embedder" in json_config["mem0"]:
                config["mem0"]["embedder"] = json_config["mem0"]["embedder"]
            if "vector_store" in json_config["mem0"]:
                config["mem0"]["vector_store"] = json_config["mem0"]["vector_store"]
        if "openmemory" in json_config:
            config["openmemory"] = json_config["openmemory"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fall back to env vars if config.json didn't provide LLM values
    if not config["mem0"]["llm"]:
        provider = os.environ.get("LLM_PROVIDER", "openai")
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        llm_config = {"model": model, "temperature": 0, "max_tokens": 4000}
        if provider == "ollama":
            llm_config["ollama_base_url"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        else:
            llm_config["api_key"] = f"env:{provider.upper()}_API_KEY"
        config["mem0"]["llm"] = {"provider": provider, "config": llm_config}

    # Fall back to env vars if config.json didn't provide embedder values
    if not config["mem0"]["embedder"]:
        provider = os.environ.get("EMBEDDER_PROVIDER", "openai")
        model = os.environ.get("EMBEDDER_MODEL", "text-embedding-3-small")
        embedder_config = {"model": model}
        if provider == "ollama":
            embedder_config["ollama_base_url"] = os.environ.get("EMBEDDER_OLLAMA_BASE_URL", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        else:
            embedder_config["api_key"] = f"env:{provider.upper()}_API_KEY"
        config["mem0"]["embedder"] = {"provider": provider, "config": embedder_config}

    return config

def get_config_from_db(db: Session, key: str = "main"):
    """Get configuration from database, falling back to defaults without persisting."""
    config = db.query(ConfigModel).filter(ConfigModel.key == key).first()

    if not config:
        # Return defaults without creating a DB row — prevents OpenAI override problem
        return get_default_configuration()

    # Ensure the config has all required sections with defaults
    config_value = config.value
    default_config = get_default_configuration()

    # Merge with defaults to ensure all required fields exist
    if "openmemory" not in config_value:
        config_value["openmemory"] = default_config["openmemory"]

    if "mem0" not in config_value:
        config_value["mem0"] = default_config["mem0"]
    else:
        if "llm" not in config_value["mem0"] or config_value["mem0"]["llm"] is None:
            config_value["mem0"]["llm"] = default_config["mem0"]["llm"]
        if "embedder" not in config_value["mem0"] or config_value["mem0"]["embedder"] is None:
            config_value["mem0"]["embedder"] = default_config["mem0"]["embedder"]
        if "vector_store" not in config_value["mem0"]:
            config_value["mem0"]["vector_store"] = default_config["mem0"]["vector_store"]

    return config_value

def save_config_to_db(db: Session, config: Dict[str, Any], key: str = "main"):
    """Save configuration to database."""
    db_config = db.query(ConfigModel).filter(ConfigModel.key == key).first()
    
    if db_config:
        db_config.value = config
        db_config.updated_at = None  # Will trigger the onupdate to set current time
    else:
        db_config = ConfigModel(key=key, value=config)
        db.add(db_config)
        
    db.commit()
    db.refresh(db_config)
    return db_config.value

@router.get("/", response_model=ConfigSchema)
async def get_configuration(db: Session = Depends(get_db)):
    """Get the current configuration."""
    config = get_config_from_db(db)
    return config

@router.put("/", response_model=ConfigSchema)
async def update_configuration(config: ConfigSchema, db: Session = Depends(get_db)):
    """Update the configuration."""
    current_config = get_config_from_db(db)
    
    # Convert to dict for processing
    updated_config = current_config.copy()
    
    # Update openmemory settings if provided
    if config.openmemory is not None:
        if "openmemory" not in updated_config:
            updated_config["openmemory"] = {}
        updated_config["openmemory"].update(config.openmemory.dict(exclude_none=True))
    
    # Update mem0 settings
    updated_config["mem0"] = config.mem0.dict(exclude_none=True)

    # Save the configuration to database
    save_config_to_db(db, updated_config)
    reset_memory_client()
    return updated_config


@router.patch("/", response_model=ConfigSchema)
async def patch_configuration(config_update: ConfigSchema, db: Session = Depends(get_db)):
    """Update parts of the configuration."""
    current_config = get_config_from_db(db)

    def deep_update(source, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                source[key] = deep_update(source[key], value)
            else:
                source[key] = value
        return source

    update_data = config_update.dict(exclude_unset=True)
    updated_config = deep_update(current_config, update_data)

    save_config_to_db(db, updated_config)
    reset_memory_client()
    return updated_config


@router.post("/reset", response_model=ConfigSchema)
async def reset_configuration(db: Session = Depends(get_db)):
    """Reset the configuration to default values."""
    try:
        # Get the default configuration with proper provider setups
        default_config = get_default_configuration()
        
        # Save it as the current configuration in the database
        save_config_to_db(db, default_config)
        reset_memory_client()
        return default_config
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to reset configuration: {str(e)}"
        )

@router.get("/mem0/llm", response_model=LLMProvider)
async def get_llm_configuration(db: Session = Depends(get_db)):
    """Get only the LLM configuration."""
    config = get_config_from_db(db)
    llm_config = config.get("mem0", {}).get("llm", {})
    return llm_config

@router.put("/mem0/llm", response_model=LLMProvider)
async def update_llm_configuration(llm_config: LLMProvider, db: Session = Depends(get_db)):
    """Update only the LLM configuration."""
    current_config = get_config_from_db(db)
    
    # Ensure mem0 key exists
    if "mem0" not in current_config:
        current_config["mem0"] = {}
    
    # Update the LLM configuration
    current_config["mem0"]["llm"] = llm_config.dict(exclude_none=True)
    
    # Save the configuration to database
    save_config_to_db(db, current_config)
    reset_memory_client()
    return current_config["mem0"]["llm"]

@router.get("/mem0/embedder", response_model=EmbedderProvider)
async def get_embedder_configuration(db: Session = Depends(get_db)):
    """Get only the Embedder configuration."""
    config = get_config_from_db(db)
    embedder_config = config.get("mem0", {}).get("embedder", {})
    return embedder_config

@router.put("/mem0/embedder", response_model=EmbedderProvider)
async def update_embedder_configuration(embedder_config: EmbedderProvider, db: Session = Depends(get_db)):
    """Update only the Embedder configuration."""
    current_config = get_config_from_db(db)
    
    # Ensure mem0 key exists
    if "mem0" not in current_config:
        current_config["mem0"] = {}
    
    # Update the Embedder configuration
    current_config["mem0"]["embedder"] = embedder_config.dict(exclude_none=True)
    
    # Save the configuration to database
    save_config_to_db(db, current_config)
    reset_memory_client()
    return current_config["mem0"]["embedder"]

@router.get("/mem0/vector_store", response_model=Optional[VectorStoreProvider])
async def get_vector_store_configuration(db: Session = Depends(get_db)):
    """Get only the Vector Store configuration."""
    config = get_config_from_db(db)
    vector_store_config = config.get("mem0", {}).get("vector_store", None)
    return vector_store_config

@router.put("/mem0/vector_store", response_model=VectorStoreProvider)
async def update_vector_store_configuration(vector_store_config: VectorStoreProvider, db: Session = Depends(get_db)):
    """Update only the Vector Store configuration."""
    current_config = get_config_from_db(db)
    
    # Ensure mem0 key exists
    if "mem0" not in current_config:
        current_config["mem0"] = {}
    
    # Update the Vector Store configuration
    current_config["mem0"]["vector_store"] = vector_store_config.dict(exclude_none=True)
    
    # Save the configuration to database
    save_config_to_db(db, current_config)
    reset_memory_client()
    return current_config["mem0"]["vector_store"]

@router.get("/openmemory", response_model=OpenMemoryConfig)
async def get_openmemory_configuration(db: Session = Depends(get_db)):
    """Get only the OpenMemory configuration."""
    config = get_config_from_db(db)
    openmemory_config = config.get("openmemory", {})
    return openmemory_config

@router.put("/openmemory", response_model=OpenMemoryConfig)
async def update_openmemory_configuration(openmemory_config: OpenMemoryConfig, db: Session = Depends(get_db)):
    """Update only the OpenMemory configuration."""
    current_config = get_config_from_db(db)
    
    # Ensure openmemory key exists
    if "openmemory" not in current_config:
        current_config["openmemory"] = {}
    
    # Update the OpenMemory configuration
    current_config["openmemory"].update(openmemory_config.dict(exclude_none=True))
    
    # Save the configuration to database
    save_config_to_db(db, current_config)
    reset_memory_client()
    return current_config["openmemory"]
