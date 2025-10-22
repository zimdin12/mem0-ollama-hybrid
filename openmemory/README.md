# OpenMemory

OpenMemory is your personal memory layer for LLMs - private, portable, and open-source. Your memories live locally, giving you complete control over your data. Build AI applications with personalized memories while keeping your data secure.

![OpenMemory](https://github.com/user-attachments/assets/3c701757-ad82-4afa-bfbe-e049c2b4320b)

## Easy Setup

### Prerequisites
- Docker
- OpenAI API Key

You can quickly run OpenMemory by running the following command:

```bash
curl -sL https://raw.githubusercontent.com/mem0ai/mem0/main/openmemory/run.sh | bash
```

You should set the `OPENAI_API_KEY` as a global environment variable:

```bash
export OPENAI_API_KEY=your_api_key
```

You can also set the `OPENAI_API_KEY` as a parameter to the script:

```bash
curl -sL https://raw.githubusercontent.com/mem0ai/mem0/main/openmemory/run.sh | OPENAI_API_KEY=your_api_key bash
```

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for backend development)
- Node.js (for frontend development)
- OpenAI API Key (required for LLM interactions, run `cp api/.env.example api/.env` then change **OPENAI_API_KEY** to yours)
  - **(FORK CHANGE)** Ollama with qwen3:4b-instruct model is also supported as an alternative to OpenAI

## Quickstart

### 1. Set Up Environment Variables

Before running the project, you need to configure environment variables for both the API and the UI.

You can do this in one of the following ways:

- **Manually**:  
  Create a `.env` file in each of the following directories:
  - `/api/.env`
  - `/ui/.env`

- **Using `.env.example` files**:  
  Copy and rename the example files:

  ```bash
  cp api/.env.example api/.env
  cp ui/.env.example ui/.env
  ```

 - **Using Makefile** (if supported):  
    Run:
  
   ```bash
   make env
   ```
- #### Example `/api/.env`

```env
OPENAI_API_KEY=sk-xxx
USER=<user-id> # The User Id you want to associate the memories with 
```
- #### Example `/ui/.env`

```env
NEXT_PUBLIC_API_URL=http://localhost:8765
NEXT_PUBLIC_USER_ID=<user-id> # Same as the user id for environment variable in api
```

### 2. Build and Run the Project
You can run the project using the following two commands:
```bash
make build # builds the mcp server and ui
make up  # runs openmemory mcp server and ui
```

After running these commands, you will have:
- OpenMemory MCP server running at: http://localhost:8765 (API documentation available at http://localhost:8765/docs)
- OpenMemory UI running at: http://localhost:3000

#### UI not working on `localhost:3000`?

If the UI does not start properly on [http://localhost:3000](http://localhost:3000), try running it manually:

```bash
cd ui
pnpm install
pnpm dev
```

### MCP Client Setup

Use the following one step command to configure OpenMemory Local MCP to a client. The general command format is as follows:

```bash
npx @openmemory/install local http://localhost:8765/mcp/<client-name>/sse/<user-id> --client <client-name>
```

Replace `<client-name>` with the desired client name and `<user-id>` with the value specified in your environment variables.


## Project Structure

- `api/` - Backend APIs + MCP server
- `ui/` - Frontend React application

## Fork Changes (Ollama + qwen3:4b Support)

**(FORK CHANGE)** This fork adds comprehensive support for using Ollama with the qwen3:4b-instruct model instead of OpenAI. The following changes have been made:

### Core Modifications

#### 1. **Essential Ollama Fix** (`api/app/utils/memory.py`)
- **Problem**: mem0 expects LLM responses in specific formats that small models like qwen3:4b don't always follow
- **Solution**: Added `_apply_essential_ollama_fix()` function that monkey-patches `OllamaLLM.generate_response()` to:
  - Parse function call syntax when returned as text: `extract_entities([...])`
  - Parse bare JSON arrays: `[{"source": "...", "relationship": "...", "destination": "..."}]`
  - Parse multiple text formats for entities (bullet lists, structured blocks, etc.)
  - Handle mixed format entities (both dicts and strings)
  - Ensure proper tool_calls format for graph extraction
  - Return strings for fact extraction, dicts for entity extraction

#### 2. **Graph Entity Extraction Fixes** (`api/fix_graph_entity_parsing.py`)
- **Problem**: qwen3:4b adds parenthetical notes and verbose explanations that break entity parsing
- **Solution**:
  - Replaced verbose mem0 prompts with concise, directive prompts
  - **Entity Extraction Prompt**: Forces tool calling with "You MUST use the extract_entities function call"
  - **Relationship Extraction Prompt**: Reduced from 210 words to 104 words with clear directives
  - Added cleaning logic for entity names and types
  - Handles mixed formats (strings and dicts) in entity arrays
  - Applied via monkey-patching `MemoryGraph._retrieve_nodes_from_data` and `MemoryGraph._establish_nodes_relations_from_data`

#### 3. **Custom System Prompts**
- **Fact Extraction**: Optimized for qwen3:4b's capabilities (stored in database)
- **Update Memory**: Simplified logic for memory updates (stored in database)
- System prompts for AI agents and chat interfaces in `api/system_prompt_*.md`

### Test Results

**Before Fork Changes:**
- Long documents (1963+ chars): 0 Neo4j graph entities ❌
- Short texts: Working ✅

**After Fork Changes:**
- Long documents (1963+ chars): 44 Neo4j graph entities ✅
- Short texts: Working ✅
- Full three-layer memory system operational:
  - SQLite: Fact storage ✅
  - Qdrant: Vector embeddings ✅
  - Neo4j: Knowledge graph ✅

### Key Files Modified

1. **`api/app/utils/memory.py`** - Essential Ollama compatibility fixes
2. **`api/fix_graph_entity_parsing.py`** - Graph extraction improvements
3. **`api/config.json`** - Optimized prompts stored in database
4. **`api/system_prompt_*.md`** - System prompts for different use cases

### Usage with Ollama

Configure in `api/.env`:
```env
# Ollama configuration
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3:4b-instruct

# Graph extraction uses same model
GRAPH_LLM_PROVIDER=ollama
GRAPH_OLLAMA_HOST=http://host.docker.internal:11434
GRAPH_OLLAMA_MODEL=qwen3:4b-instruct
```

### Integration Support

**(FORK CHANGE)** This fork is ready for integration with:
- **Goose**: MCP server fully compatible
- **Open WebUI**: API endpoints and memory management ready
- **Other MCP clients**: Standard MCP protocol implementation

### Technical Details

For a comprehensive breakdown of all prompt changes and parsing logic, see:
- `api/PROMPT_CHANGES_SUMMARY.md` - Detailed before/after prompt comparison
- `api/ENHANCED_MEMORY_FEATURES.md` - Feature documentation

### Notes

- All changes are backward compatible with OpenAI models
- The parsing fallbacks ensure robustness across different LLM behaviors
- Changes are applied via monkey-patching, requiring no modifications to the mem0 library
- System uses intelligent parsing with multiple fallback strategies

## Contributing

We are a team of developers passionate about the future of AI and open-source software. With years of experience in both fields, we believe in the power of community-driven development and are excited to build tools that make AI more accessible and personalized.

We welcome all forms of contributions:
- Bug reports and feature requests
- Documentation improvements
- Code contributions
- Testing and feedback
- Community support

How to contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b openmemory/feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin openmemory/feature/amazing-feature`)
5. Open a Pull Request

Join us in building the future of AI memory management! Your contributions help make OpenMemory better for everyone.
