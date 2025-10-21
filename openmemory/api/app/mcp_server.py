"""
MCP Server for OpenMemory with resilient memory client handling.

This module implements an MCP (Model Context Protocol) server that provides
memory operations for OpenMemory. The memory client is initialized lazily
to prevent server crashes when external dependencies (like Ollama) are
unavailable. If the memory client cannot be initialized, the server will
continue running with limited functionality and appropriate error messages.

Key features:
- Lazy memory client initialization
- Graceful error handling for unavailable dependencies
- Fallback to database-only mode when vector store is unavailable
- Proper logging for debugging connection issues
- Environment variable parsing for API keys
"""

import contextvars
import datetime
import json
import logging
import uuid

from app.database import SessionLocal
from app.models import Memory, MemoryAccessLog, MemoryState, MemoryStatusHistory
from app.utils.db import get_user_and_app
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions
from app.utils.enhanced_memory import enhanced_memory_manager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

# Load environment variables
load_dotenv()

# Initialize MCP
mcp = FastMCP("mem0-mcp-server")

# Don't initialize memory client at import time - do it lazily when needed
def get_memory_client_safe():
    """Get memory client with error handling. Returns None if client cannot be initialized."""
    try:
        return get_memory_client()
    except Exception as e:
        logging.warning(f"Failed to get memory client: {e}")
        return None

# Context variables for user_id and client_name
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")

# Create a router for MCP endpoints
mcp_router = APIRouter(prefix="/mcp")

# Initialize SSE transport
sse = SseServerTransport("/mcp/messages/")

@mcp.tool(description="Intelligently add new memories by checking existing knowledge and only storing truly new information. Provides detailed feedback about what was added, updated, or already known.")
async def add_memories(text: str) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Check if app is active
            if not app.is_active:
                return f"Error: App {app.name} is currently paused on OpenMemory. Cannot create new memories."

            # Use enhanced memory manager for smart addition
            addition_result = enhanced_memory_manager.smart_add_memory(
                text, 
                uid, 
                metadata={
                    "source_app": "openmemory",
                    "mcp_client": client_name,
                }
            )
            
            response = {
                "status": addition_result.status,
                "summary": addition_result.summary,
                "added_memories": addition_result.added_memories,
                "related_memories": addition_result.related_memories[:3],
                "insights": f"Processed {len(addition_result.added_memories)} new memories, "
                           f"found {len(addition_result.related_memories)} related memories, "
                           f"skipped {len(addition_result.skipped_facts)} duplicate facts."
            }

            # Process added memories and update database
            for memory_data in addition_result.added_memories:
                if 'id' in memory_data:
                    memory_id = uuid.UUID(memory_data['id'])
                    
                    # Check if memory already exists
                    memory = db.query(Memory).filter(Memory.id == memory_id).first()
                    if not memory:
                        memory = Memory(
                            id=memory_id,
                            user_id=user.id,
                            app_id=app.id,
                            content=memory_data.get('memory', text),
                            state=MemoryState.active
                        )
                        db.add(memory)
                        
                        # Create history entry (for new memories, old_state should be the same as new_state initially)
                        history = MemoryStatusHistory(
                            memory_id=memory_id,
                            changed_by=user.id,
                            old_state=MemoryState.active,  # For new memories, treat as if it was always active
                            new_state=MemoryState.active
                        )
                        db.add(history)

            db.commit()
            return json.dumps(response, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error adding to memory: {e}")
        return f"Error adding to memory: {e}"


@mcp.tool(description="Perform hybrid search across all memory systems (vector, graph, temporal) to find relevant information. Returns comprehensive results with relationships and context.")
async def search_memory(query: str) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Perform hybrid search using enhanced memory manager
            search_results = enhanced_memory_manager.hybrid_search(query, uid, limit=15)
            
            # Filter results based on permissions
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = {str(memory.id) for memory in user_memories if check_memory_access_permissions(db, memory, app.id)}

            filtered_results = []
            for result in search_results:
                if result.id in accessible_memory_ids or not result.id:  # Allow results without IDs (from graph/temporal)
                    filtered_results.append({
                        "id": result.id,
                        "memory": result.content,
                        "score": result.score,
                        "source": result.source,
                        "metadata": result.metadata,
                        "relationships": result.relationships,
                        "created_at": result.created_at.isoformat() if result.created_at and hasattr(result.created_at, 'isoformat') else str(result.created_at) if result.created_at else None,
                        "updated_at": result.updated_at.isoformat() if result.updated_at and hasattr(result.updated_at, 'isoformat') else str(result.updated_at) if result.updated_at else None,
                    })

            # Log access for memories with IDs
            for result in filtered_results:
                if result.get("id"):
                    try:
                        access_log = MemoryAccessLog(
                            memory_id=uuid.UUID(result["id"]),
                            app_id=app.id,
                            access_type="hybrid_search",
                            metadata_={
                                "query": query,
                                "score": result.get("score"),
                                "source": result.get("source"),
                            },
                        )
                        db.add(access_log)
                    except ValueError:
                        # Skip invalid UUIDs
                        pass
            
            db.commit()

            # Prepare comprehensive response
            response = {
                "query": query,
                "total_results": len(filtered_results),
                "results": filtered_results,
                "search_strategy": "hybrid (vector + graph + temporal)",
                "sources_found": list(set(r["source"] for r in filtered_results)),
                "insights": [
                    f"Found {len(filtered_results)} relevant memories",
                    f"Search covered {len(set(r['source'] for r in filtered_results))} different memory dimensions"
                ]
            }

            return json.dumps(response, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(e)
        return f"Error searching memory: {e}"


@mcp.tool(description="List all memories in the user's memory")
async def list_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Get all memories
            memories = memory_client.get_all(user_id=uid)
            filtered_memories = []

            # Filter memories based on permissions
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            if isinstance(memories, dict) and 'results' in memories:
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        if memory_id in accessible_memory_ids:
                            # Create access log entry
                            access_log = MemoryAccessLog(
                                memory_id=memory_id,
                                app_id=app.id,
                                access_type="list",
                                metadata_={
                                    "hash": memory_data.get('hash')
                                }
                            )
                            db.add(access_log)
                            filtered_memories.append(memory_data)
                db.commit()
            else:
                for memory in memories:
                    memory_id = uuid.UUID(memory['id'])
                    memory_obj = db.query(Memory).filter(Memory.id == memory_id).first()
                    if memory_obj and check_memory_access_permissions(db, memory_obj, app.id):
                        # Create access log entry
                        access_log = MemoryAccessLog(
                            memory_id=memory_id,
                            app_id=app.id,
                            access_type="list",
                            metadata_={
                                "hash": memory.get('hash')
                            }
                        )
                        db.add(access_log)
                        filtered_memories.append(memory)
                db.commit()
            return json.dumps(filtered_memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memories: {e}")
        return f"Error getting memories: {e}"


@mcp.tool(description="Delete all memories in the user's memory")
async def delete_all_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            # delete the accessible memories only
            for memory_id in accessible_memory_ids:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in accessible_memory_ids:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                # Update memory state
                memory.state = MemoryState.deleted
                memory.deleted_at = now

                # Create history entry
                history = MemoryStatusHistory(
                    memory_id=memory_id,
                    changed_by=user.id,
                    old_state=MemoryState.active,
                    new_state=MemoryState.deleted
                )
                db.add(history)

                # Create access log entry
                access_log = MemoryAccessLog(
                    memory_id=memory_id,
                    app_id=app.id,
                    access_type="delete_all",
                    metadata_={"operation": "bulk_delete"}
                )
                db.add(access_log)

            db.commit()
            return "Successfully deleted all memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp.tool(description="Comprehensive memory processing like human memory - automatically extract, store, relate, and provide context from conversation turns. Pass both user message and LLM response for intelligent memory management.")
async def handle_conversation(user_message: str, llm_response: str) -> str:
    """
    Comprehensive memory handling that processes both user and LLM messages
    like human memory - extracting, storing, relating, and providing context
    """
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Check if app is active
            if not app.is_active:
                return f"Error: App {app.name} is currently paused on OpenMemory."

            # Use enhanced memory manager for comprehensive processing
            result = enhanced_memory_manager.comprehensive_memory_handle(
                user_message, 
                llm_response, 
                uid
            )
            
            # Process any new memories that were added
            for memory_update in result.get("memory_updates", []):
                addition_result = memory_update.get("result", {})
                for memory_data in addition_result.get("added_memories", []):
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        
                        # Check if memory already exists
                        memory = db.query(Memory).filter(Memory.id == memory_id).first()
                        if not memory:
                            memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=memory_data.get('memory', memory_update.get('content', '')),
                                state=MemoryState.active
                            )
                            db.add(memory)
                            
                            # Create history entry (for new memories, old_state should be the same as new_state initially)
                            history = MemoryStatusHistory(
                                memory_id=memory_id,
                                changed_by=user.id,
                                old_state=MemoryState.active,  # For new memories, treat as if it was always active
                                new_state=MemoryState.active
                            )
                            db.add(history)

            db.commit()
            
            # Prepare comprehensive response
            response = {
                "processing_status": result.get("status", "processed"),
                "memory_operations": {
                    "extracted_content": len(result.get("memory_updates", [])),
                    "related_memories_found": len(result.get("related_context", [])),
                    "patterns_identified": result.get("patterns", []),
                    "insights_generated": result.get("insights", [])
                },
                "contextual_information": {
                    "relevant_memories": result.get("related_context", [])[:5],  # Limit for readability
                    "conversation_patterns": result.get("patterns", []),
                    "memory_insights": result.get("insights", [])
                },
                "memory_updates": result.get("memory_updates", []),
                "summary": f"Processed conversation turn: extracted {len(result.get('memory_updates', []))} memory items, "
                          f"found {len(result.get('related_context', []))} related memories, "
                          f"identified {len(result.get('patterns', []))} conversation patterns."
            }
            
            return json.dumps(response, indent=2)
            
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in comprehensive memory handling: {e}")
        return f"Error processing conversation: {e}"


@mcp.tool(description="Get memories related to specific entities or topics, with relationship traversal and contextual connections.")
async def get_related_memories(topic: str, max_depth: int = 2) -> str:
    """
    Find memories related to a specific topic with relationship traversal
    """
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        # Perform enhanced search for the topic
        search_results = enhanced_memory_manager.hybrid_search(topic, uid, limit=20)
        
        # Group results by source and relationships
        related_memories = {
            "direct_matches": [],
            "relationship_connections": [],
            "temporal_context": [],
            "topic_summary": topic
        }
        
        for result in search_results:
            memory_data = {
                "id": result.id,
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata,
                "relationships": result.relationships
            }
            
            if result.source == "vector":
                related_memories["direct_matches"].append(memory_data)
            elif result.source == "graph":
                related_memories["relationship_connections"].append(memory_data)
            elif result.source == "temporal":
                related_memories["temporal_context"].append(memory_data)
        
        # Add insights about the topic
        total_memories = len(search_results)
        related_memories["insights"] = [
            f"Found {total_memories} memories related to '{topic}'",
            f"Direct semantic matches: {len(related_memories['direct_matches'])}",
            f"Relationship connections: {len(related_memories['relationship_connections'])}",
            f"Temporal context: {len(related_memories['temporal_context'])}"
        ]
        
        return json.dumps(related_memories, indent=2)
        
    except Exception as e:
        logging.exception(f"Error getting related memories: {e}")
        return f"Error getting related memories: {e}"


@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for a specific user and client"""
    # Extract user_id and client_name from path parameters
    uid = request.path_params.get("user_id")
    user_token = user_id_var.set(uid or "")
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")

    try:
        # Handle SSE connection
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    finally:
        # Clean up context variables
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)


@mcp_router.post("/messages/")
async def handle_get_message(request: Request):
    return await handle_post_message(request)


@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_post_message(request: Request):
    return await handle_post_message(request)

async def handle_post_message(request: Request):
    """Handle POST messages for SSE"""
    try:
        body = await request.body()

        # Create a simple receive function that returns the body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Create a simple send function that does nothing
        async def send(message):
            return {}

        # Call handle_post_message with the correct arguments
        await sse.handle_post_message(request.scope, receive, send)

        # Return a success response
        return {"status": "ok"}
    finally:
        pass

def setup_mcp_server(app: FastAPI):
    """Setup MCP server with the FastAPI application"""
    mcp._mcp_server.name = "mem0-mcp-server"

    # Include MCP router in the FastAPI app
    app.include_router(mcp_router)
