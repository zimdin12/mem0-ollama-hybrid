import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from app.database import get_db
from app.models import (
    AccessControl,
    App,
    Category,
    Memory,
    MemoryAccessLog,
    MemoryState,
    MemoryStatusHistory,
    User,
)
from app.schemas import MemoryResponse
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlalchemy import paginate as sqlalchemy_paginate
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

# Add imports for Qdrant sync
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])


def get_memory_or_404(db: Session, memory_id: UUID) -> Memory:
    memory = db.query(Memory).filter(Memory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


def update_memory_state(db: Session, memory_id: UUID, new_state: MemoryState, user_id: UUID):
    memory = get_memory_or_404(db, memory_id)
    old_state = memory.state

    # Update memory state
    memory.state = new_state
    if new_state == MemoryState.archived:
        memory.archived_at = datetime.now(UTC)
    elif new_state == MemoryState.deleted:
        memory.deleted_at = datetime.now(UTC)

    # Record state change
    history = MemoryStatusHistory(
        memory_id=memory_id,
        changed_by=user_id,
        old_state=old_state,
        new_state=new_state
    )
    db.add(history)
    db.commit()
    return memory


def get_accessible_memory_ids(db: Session, app_id: UUID) -> Set[UUID]:
    """
    Get the set of memory IDs that the app has access to based on app-level ACL rules.
    Returns all memory IDs if no specific restrictions are found.
    """
    # Get app-level access controls
    app_access = db.query(AccessControl).filter(
        AccessControl.subject_type == "app",
        AccessControl.subject_id == app_id,
        AccessControl.object_type == "memory"
    ).all()

    # If no app-level rules exist, return None to indicate all memories are accessible
    if not app_access:
        return None

    # Initialize sets for allowed and denied memory IDs
    allowed_memory_ids = set()
    denied_memory_ids = set()

    # Process app-level rules
    for rule in app_access:
        if rule.effect == "allow":
            if rule.object_id:  # Specific memory access
                allowed_memory_ids.add(rule.object_id)
            else:  # All memories access
                return None  # All memories allowed
        elif rule.effect == "deny":
            if rule.object_id:  # Specific memory denied
                denied_memory_ids.add(rule.object_id)
            else:  # All memories denied
                return set()  # No memories accessible

    # Remove denied memories from allowed set
    if allowed_memory_ids:
        allowed_memory_ids -= denied_memory_ids

    return allowed_memory_ids


# Helper function to query graph (mem0 0.1.118 compatible)
def query_graph(user_id: str, query_str: str, params: dict = None) -> List[Dict]:
    """Execute a Cypher query on the graph database"""
    try:
        memory_client = get_memory_client()
        if not memory_client or not hasattr(memory_client, 'graph'):
            return []
        
        graph = memory_client.graph
        params = params or {}
        params['user_id'] = user_id
        
        # mem0 0.1.118 uses graph.graph.query() instead of driver.session()
        if hasattr(graph, 'graph') and hasattr(graph.graph, 'query'):
            result = graph.graph.query(query_str, params)
            # Result format depends on the query, try to normalize it
            if isinstance(result, list):
                return result
            return [result] if result else []
        return []
    except Exception as e:
        logging.error(f"Graph query error: {e}")
        return []


# Get entities for a memory from graph
def get_memory_entities_from_graph(memory_id: UUID, user_id: str) -> List[Dict]:
    """Get entities extracted from a specific memory"""
    query = """
    MATCH (n)-[:MENTIONED_IN]->(m:Memory {id: $memory_id})
    WHERE m.user_id = $user_id
    RETURN n.name as name, labels(n) as types, properties(n) as properties
    """
    return query_graph(user_id, query, {"memory_id": str(memory_id)})


# Get related memories through graph relationships
def get_related_memories_from_graph(memory_id: UUID, user_id: str, limit: int = 10) -> List[str]:
    """Find related memories through shared entities"""
    query = """
    MATCH (entity)-[:MENTIONED_IN]->(m1:Memory {id: $memory_id})
    MATCH (entity)-[:MENTIONED_IN]->(m2:Memory)
    WHERE m1.user_id = $user_id AND m2.user_id = $user_id AND m1.id <> m2.id
    RETURN DISTINCT m2.id as related_memory_id, count(entity) as shared_entities
    ORDER BY shared_entities DESC
    LIMIT $limit
    """
    results = query_graph(user_id, query, {"memory_id": str(memory_id), "limit": limit})
    return [r["related_memory_id"] for r in results]


# List all memories with filtering
@router.get("/", response_model=Page[MemoryResponse])
async def list_memories(
    user_id: str,
    app_id: Optional[UUID] = None,
    from_date: Optional[int] = Query(
        None,
        description="Filter memories created after this date (timestamp)",
        examples=[1718505600]
    ),
    to_date: Optional[int] = Query(
        None,
        description="Filter memories created before this date (timestamp)",
        examples=[1718505600]
    ),
    categories: Optional[str] = None,
    params: Params = Depends(),
    search_query: Optional[str] = None,
    sort_column: Optional[str] = Query(None, description="Column to sort by (memory, categories, app_name, created_at)"),
    sort_direction: Optional[str] = Query(None, description="Sort direction (asc or desc)"),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Build base query
    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
        Memory.state != MemoryState.archived,
        Memory.content.ilike(f"%{search_query}%") if search_query else True
    )

    # Apply filters
    if app_id:
        query = query.filter(Memory.app_id == app_id)

    if from_date:
        from_datetime = datetime.fromtimestamp(from_date, tz=UTC)
        query = query.filter(Memory.created_at >= from_datetime)

    if to_date:
        to_datetime = datetime.fromtimestamp(to_date, tz=UTC)
        query = query.filter(Memory.created_at <= to_datetime)

    # Add joins for app and categories after filtering
    query = query.outerjoin(App, Memory.app_id == App.id)
    query = query.outerjoin(Memory.categories)

    # Apply category filter if provided
    if categories:
        category_list = [c.strip() for c in categories.split(",")]
        query = query.filter(Category.name.in_(category_list))

    # Apply sorting if specified
    if sort_column:
        sort_field = getattr(Memory, sort_column, None)
        if sort_field:
            query = query.order_by(sort_field.desc()) if sort_direction == "desc" else query.order_by(sort_field.asc())

    # Add eager loading for app and categories
    query = query.options(
        joinedload(Memory.app),
        joinedload(Memory.categories)
    ).distinct(Memory.id)
    # Get paginated results with transformer
    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
            if check_memory_access_permissions(db, memory, app_id)
        ]
    )


# Get all categories
@router.get("/categories")
async def get_categories(
    user_id: str,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get unique categories associated with the user's memories
    memories = db.query(Memory).filter(Memory.user_id == user.id, Memory.state != MemoryState.deleted, Memory.state != MemoryState.archived).all()
    categories = [category for memory in memories for category in memory.categories]
    unique_categories = list(set(categories))

    return {
        "categories": unique_categories,
        "total": len(unique_categories)
    }


class CreateMemoryRequest(BaseModel):
    user_id: str
    text: Optional[str] = None
    messages: Optional[List[dict]] = None
    metadata: dict = {}
    infer: bool = True
    app: str = "openmemory"


# Create new memory
@router.post("/")
async def create_memory(
    request: CreateMemoryRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Extract text from messages if text is not provided
    if not request.text and request.messages:
        request.text = " ".join([
            msg.get("content", "") 
            for msg in request.messages 
            if msg.get("content")
        ])
    
    # Validate that we have text to process
    if not request.text:
        raise HTTPException(
            status_code=400, 
            detail="Either 'text' or 'messages' must be provided"
        )
        
    # Get or create app
    app_obj = db.query(App).filter(App.name == request.app,
                                   App.owner_id == user.id).first()
    if not app_obj:
        app_obj = App(name=request.app, owner_id=user.id)
        db.add(app_obj)
        db.commit()
        db.refresh(app_obj)

    # Check if app is active
    if not app_obj.is_active:
        raise HTTPException(status_code=403, detail=f"App {request.app} is currently paused on OpenMemory. Cannot create new memories.")

    logging.info(f"Creating memory for user_id: {request.user_id} with app: {request.app}")
    
    # Track processed memory IDs to prevent duplicates across all processing paths
    processed_memory_ids = set()
    
    # Try to get memory client safely
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise Exception("Memory client is not available")
    except Exception as client_error:
        logging.warning(f"Memory client unavailable: {client_error}. Creating memory in database only.")
        return {"error": str(client_error)}

    # Try automatic memory extraction first
    try:
        qdrant_response = memory_client.add(
            request.text,
            user_id=request.user_id,
            metadata={
                "source_app": "openmemory",
                "mcp_client": request.app,
            },
            infer=request.infer
        )
        
        logging.info(f"Memory add response: {qdrant_response}")
        
        # Check if entities were extracted
        if 'relations' in qdrant_response:
            logging.info(f"Entities extracted: {len(qdrant_response['relations'].get('added_entities', []))}")
            if qdrant_response['relations'].get('added_entities'):
                logging.info(f"Added entities: {qdrant_response['relations']['added_entities']}")
        
        # FALLBACK: If mem0 returns empty results, store the text directly with chunking
        if isinstance(qdrant_response, dict) and 'results' in qdrant_response:
            if not qdrant_response['results']:
                logging.warning("mem0 returned empty results - using enhanced fallback: chunking and storing in both vector and SQL")
                
                # Import chunking utilities
                from uuid import uuid4
                import re
                
                def chunk_text(text, target_chunk_size=600, max_chunk_size=1000):
                    """Dynamically chunk text based on content structure and optimal size for LLM processing"""
                    if len(text) <= target_chunk_size:
                        return [text]
                    
                    # Split by paragraphs first, then by sentences if needed
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    
                    chunks = []
                    current_chunk = ""
                    
                    for paragraph in paragraphs:
                        # If this paragraph alone is too long, split it by sentences
                        if len(paragraph) > max_chunk_size:
                            # Split by sentences (rough approximation)
                            sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
                            
                            for sentence in sentences:
                                if len(current_chunk) + len(sentence) > target_chunk_size and current_chunk:
                                    chunks.append(current_chunk.strip())
                                    current_chunk = sentence
                                else:
                                    current_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
                        else:
                            # Normal paragraph processing - use target size for better chunking
                            if len(current_chunk) + len(paragraph) > target_chunk_size and current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = paragraph
                            else:
                                current_chunk = current_chunk + '\n\n' + paragraph if current_chunk else paragraph
                    
                    # Add the last chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    # Post-process: split oversized chunks by word boundaries
                    final_chunks = []
                    for chunk in chunks:
                        if len(chunk) <= max_chunk_size:
                            final_chunks.append(chunk)
                        else:
                            # Split oversized chunk by words
                            words = chunk.split()
                            temp_chunk = ""
                            
                            for word in words:
                                if len(temp_chunk) + len(word) + 1 > target_chunk_size and temp_chunk:
                                    final_chunks.append(temp_chunk.strip())
                                    temp_chunk = word
                                else:
                                    temp_chunk = temp_chunk + ' ' + word if temp_chunk else word
                            
                            if temp_chunk.strip():
                                final_chunks.append(temp_chunk.strip())
                    
                    # Ensure we don't return empty chunks
                    final_chunks = [chunk for chunk in final_chunks if chunk.strip()]
                    
                    logging.info(f"Dynamic chunking: {len(text)} chars -> {len(final_chunks)} chunks (target: {target_chunk_size}, max: {max_chunk_size})")
                    return final_chunks if final_chunks else [text]
                
                # Chunk the text for better processing
                text_chunks = chunk_text(request.text)
                logging.info(f"Split long text into {len(text_chunks)} chunks for processing")
                
                created_memories = []
                
                # Process each chunk through mem0 individually
                for i, chunk in enumerate(text_chunks):
                    try:
                        chunk_result = memory_client.add(
                            chunk,
                            user_id=request.user_id,
                            metadata={
                                **request.metadata,
                                "chunk_index": i,
                                "total_chunks": len(text_chunks),
                                "is_chunked_document": True,
                                "original_length": len(request.text)
                            }
                        )
                        
                        # Process chunk results
                        if chunk_result and chunk_result.get('results'):
                            for result in chunk_result['results']:
                                if 'id' in result:
                                    # Convert string ID from mem0 to UUID for database
                                    memory_id = UUID(result['id']) if isinstance(result['id'], str) else result['id']
                                    
                                    # Create memory in database
                                    memory = Memory(
                                        id=memory_id,
                                        user_id=user.id,
                                        app_id=app_obj.id,
                                        content=result.get('memory', chunk),
                                        metadata_={
                                            **request.metadata,
                                            "chunk_index": i,
                                            "total_chunks": len(text_chunks),
                                            "is_chunked_document": True
                                        },
                                        state=MemoryState.active
                                    )
                                    db.add(memory)
                                    created_memories.append(memory)
                                    
                                    # Create history entry
                                    history = MemoryStatusHistory(
                                        memory_id=memory_id,
                                        changed_by=user.id,
                                        old_state=MemoryState.active,
                                        new_state=MemoryState.active
                                    )
                                    db.add(history)
                        
                        # If chunk returned no results, use direct storage fallback
                        if not chunk_result.get('results'):
                            logging.info(f"Chunk {i+1} returned no results - using direct vector storage")
                            
                            memory_id = uuid4()
                            
                            # Store in database only if not already processed
                            if str(memory_id) not in processed_memory_ids:
                                memory = Memory(
                                    id=memory_id,
                                    user_id=user.id,
                                    app_id=app_obj.id,
                                    content=chunk,
                                    metadata_={
                                        **request.metadata,
                                        "chunk_index": i,
                                        "total_chunks": len(text_chunks),
                                        "is_chunked_document": True,
                                        "storage_method": "direct_vector_no_results"
                                    },
                                    state=MemoryState.active
                                )
                                db.add(memory)
                                created_memories.append(memory)
                                processed_memory_ids.add(str(memory_id))
                                
                                # Create history entry
                                history = MemoryStatusHistory(
                                    memory_id=memory_id,
                                    changed_by=user.id,
                                    old_state=MemoryState.active,
                                    new_state=MemoryState.active
                                )
                                db.add(history)
                            
                            # Store directly in Qdrant for semantic search - use improved method
                            qdrant_stored = False
                            try:
                                if hasattr(memory_client, 'vector_store') and memory_client.vector_store and hasattr(memory_client, 'embedder'):
                                    # Get embedding for the chunk
                                    embedding = memory_client.embedder.embed_query(chunk)
                                    logging.info(f"Created embedding for chunk {i+1}: {len(embedding)} dimensions")
                                    
                                    # Try using the client's upsert method directly
                                    if hasattr(memory_client.vector_store, 'client') and hasattr(memory_client.vector_store.client, 'upsert'):
                                        from qdrant_client.models import PointStruct
                                        
                                        points = [PointStruct(
                                            id=f"{memory_id}_chunk_{i}",
                                            vector=embedding,
                                            payload={
                                                "user_id": request.user_id,
                                                "chunk_index": i,
                                                "total_chunks": len(text_chunks),
                                                "storage_method": "direct_no_results",
                                                "memory_id": str(memory_id),
                                                "text": chunk
                                            }
                                        )]
                                        
                                        # Get collection name from vector store
                                        collection_name = getattr(memory_client.vector_store, 'collection_name', 'mem0')
                                        
                                        result = memory_client.vector_store.client.upsert(
                                            collection_name=collection_name,
                                            points=points
                                        )
                                        
                                        logging.info(f"✅ Stored chunk {i+1} in Qdrant using client.upsert: {result}")
                                        qdrant_stored = True
                                        
                                    elif hasattr(memory_client.vector_store, 'insert'):
                                        # Fallback to insert method
                                        memory_client.vector_store.insert(
                                            vectors=[embedding],
                                            payloads=[{
                                                "user_id": request.user_id,
                                                "chunk_index": i,
                                                "total_chunks": len(text_chunks),
                                                "storage_method": "direct_no_results",
                                                "memory_id": str(memory_id),
                                                "text": chunk
                                            }],
                                            ids=[f"{memory_id}_chunk_{i}"]
                                        )
                                        logging.info(f"✅ Stored chunk {i+1} in Qdrant using insert method")
                                        qdrant_stored = True
                                        
                            except Exception as vector_error:
                                logging.error(f"❌ Failed to store chunk {i+1} in Qdrant: {vector_error}")
                                import traceback
                                logging.error(f"Qdrant storage traceback: {traceback.format_exc()}")
                            
                            if not qdrant_stored:
                                logging.warning(f"⚠️ Chunk {i+1} stored in SQLite but NOT in Qdrant - semantic search will not find this chunk")
                        
                        logging.info(f"Processed chunk {i+1}/{len(text_chunks)}: {len(chunk_result.get('results', []))} memories")
                        
                    except Exception as chunk_error:
                        logging.error(f"Error processing chunk {i}: {chunk_error}")
                        
                        # Enhanced fallback: store chunk directly in both SQLite AND Qdrant
                        logging.info(f"Chunk {i} failed mem0 processing - using direct vector storage fallback")
                        
                        memory_id = uuid4()
                        
                        # Store in database first only if not already processed
                        if str(memory_id) not in processed_memory_ids:
                            memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app_obj.id,
                                content=chunk,
                                metadata_={
                                    **request.metadata,
                                    "chunk_index": i,
                                    "total_chunks": len(text_chunks),
                                    "is_chunked_document": True,
                                    "storage_method": "direct_vector_fallback"
                                },
                                state=MemoryState.active
                            )
                            db.add(memory)
                            created_memories.append(memory)
                            processed_memory_ids.add(str(memory_id))
                        
                        # Create history entry
                        history = MemoryStatusHistory(
                            memory_id=memory_id,
                            changed_by=user.id,
                            old_state=MemoryState.active,
                            new_state=MemoryState.active
                        )
                        db.add(history)
                        
                        # ALSO store directly in Qdrant for semantic search
                        qdrant_stored = False
                        try:
                            # Access the vector store directly to store this chunk
                            if hasattr(memory_client, 'vector_store') and memory_client.vector_store and hasattr(memory_client, 'embedder'):
                                # Get embedding for the chunk
                                embedding = memory_client.embedder.embed_query(chunk)
                                logging.info(f"Created embedding for chunk {i} exception fallback: {len(embedding)} dimensions")
                                
                                # Try using the client's upsert method directly first
                                if hasattr(memory_client.vector_store, 'client') and hasattr(memory_client.vector_store.client, 'upsert'):
                                    from qdrant_client.models import PointStruct
                                    
                                    points = [PointStruct(
                                        id=f"{memory_id}_chunk_{i}",
                                        vector=embedding,
                                        payload={
                                            "user_id": request.user_id,
                                            "chunk_index": i,
                                            "total_chunks": len(text_chunks),
                                            "storage_method": "direct_fallback",
                                            "memory_id": str(memory_id),
                                            "text": chunk
                                        }
                                    )]
                                    
                                    # Get collection name from vector store
                                    collection_name = getattr(memory_client.vector_store, 'collection_name', 'mem0')
                                    
                                    result = memory_client.vector_store.client.upsert(
                                        collection_name=collection_name,
                                        points=points
                                    )
                                    
                                    logging.info(f"✅ Stored chunk {i} in Qdrant using client.upsert (exception fallback): {result}")
                                    qdrant_stored = True
                                    
                                elif hasattr(memory_client.vector_store, 'insert'):
                                    # Fallback to insert method
                                    memory_client.vector_store.insert(
                                        vectors=[embedding],
                                        payloads=[{
                                            "user_id": request.user_id,
                                            "chunk_index": i,
                                            "total_chunks": len(text_chunks),
                                            "storage_method": "direct_fallback",
                                            "memory_id": str(memory_id),
                                            "text": chunk
                                        }],
                                        ids=[f"{memory_id}_chunk_{i}"]
                                    )
                                    logging.info(f"✅ Stored chunk {i} in Qdrant using insert method (exception fallback)")
                                    qdrant_stored = True
                                    
                        except Exception as vector_error:
                            logging.error(f"❌ Failed to store chunk {i} in Qdrant (exception fallback): {vector_error}")
                            import traceback
                            logging.error(f"Qdrant storage traceback: {traceback.format_exc()}")
                        
                        if not qdrant_stored:
                            logging.warning(f"⚠️ Chunk {i} stored in SQLite but NOT in Qdrant - semantic search will not find this chunk")
                
                db.commit()
                
                logging.info(f"✅ Enhanced fallback complete: {len(created_memories)} memories stored from {len(text_chunks)} chunks")
                
                # Return the first memory as representative
                return created_memories[0] if created_memories else None
            
            # Process normal results
            created_memories = []
            
            for result in qdrant_response['results']:
                if result['event'] in ['ADD', 'UPDATE']:
                    # Ensure ID is always a UUID object
                    memory_id = UUID(result['id']) if isinstance(result['id'], str) else result['id']
                    
                    # Skip if already processed to prevent duplicates
                    if str(memory_id) in processed_memory_ids:
                        continue
                        
                    existing_memory = db.query(Memory).filter(Memory.id == memory_id).first()
                    
                    if existing_memory:
                        existing_memory.state = MemoryState.active
                        existing_memory.content = result['memory']
                        memory = existing_memory
                    else:
                        memory = Memory(
                            id=memory_id,
                            user_id=user.id,
                            app_id=app_obj.id,
                            content=result['memory'],
                            metadata_=request.metadata,
                            state=MemoryState.active
                        )
                        db.add(memory)
                    
                    history = MemoryStatusHistory(
                        memory_id=memory_id,
                        changed_by=user.id,
                        old_state=MemoryState.deleted if existing_memory else MemoryState.deleted,
                        new_state=MemoryState.active
                    )
                    db.add(history)
                    
                    created_memories.append(memory)
                    processed_memory_ids.add(str(memory_id))
            
            if created_memories:
                db.commit()
                for memory in created_memories:
                    db.refresh(memory)
                
                return created_memories[0]
                
    except Exception as qdrant_error:
        logging.error(f"Qdrant operation failed: {qdrant_error}")
        import traceback
        traceback.print_exc()
        return {"error": str(qdrant_error)}


# Enhanced get memory with graph entities
@router.get("/{memory_id}")
async def get_memory(
    memory_id: UUID,
    include_entities: bool = Query(False, description="Include extracted entities from graph"),
    db: Session = Depends(get_db)
):
    memory = get_memory_or_404(db, memory_id)
    
    response = {
        "id": memory.id,
        "text": memory.content,
        "created_at": int(memory.created_at.timestamp()),
        "state": memory.state.value,
        "app_id": memory.app_id,
        "app_name": memory.app.name if memory.app else None,
        "categories": [category.name for category in memory.categories],
        "metadata_": memory.metadata_
    }
    
    if include_entities:
        user = db.query(User).filter(User.id == memory.user_id).first()
        if user:
            entities = get_memory_entities_from_graph(memory_id, user.user_id)
            response["entities"] = [
                {
                    "name": e["name"],
                    "type": e["types"][0] if e["types"] else "Unknown"
                }
                for e in entities
            ]
    
    return response


class DeleteMemoriesRequest(BaseModel):
    memory_ids: List[UUID]
    user_id: str


@router.delete("/")
async def delete_memories(
    request: DeleteMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get memory client to delete from vector store
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(
                status_code=503,
                detail="Memory client is not available"
            )
    except HTTPException:
        raise
    except Exception as client_error:
        logging.error(f"Memory client initialization failed: {client_error}")
        raise HTTPException(
            status_code=503,
            detail=f"Memory service unavailable: {str(client_error)}"
        )

    # Delete from vector store then mark as deleted in database
    for memory_id in request.memory_ids:
        try:
            memory_client.delete(str(memory_id))
        except Exception as delete_error:
            logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

        update_memory_state(db, memory_id, MemoryState.deleted, user.id)

    try:
        memory_client = get_memory_client()
        if memory_client:
            for memory_id in request.memory_ids:
                try:
                    memory_client.delete(memory_id=str(memory_id), user_id=request.user_id)
                except Exception as e:
                    logging.warning(f"Failed to delete memory {memory_id} from Qdrant: {e}")
    except Exception as e:
        logging.warning(f"Failed to get memory client for deletion: {e}")
    
    return {"message": f"Successfully deleted {len(request.memory_ids)} memories from both stores"}


@router.post("/actions/archive")
async def archive_memories(
    memory_ids: List[UUID],
    user_id: UUID,
    db: Session = Depends(get_db)
):
    for memory_id in memory_ids:
        update_memory_state(db, memory_id, MemoryState.archived, user_id)
    return {"message": f"Successfully archived {len(memory_ids)} memories"}


class PauseMemoriesRequest(BaseModel):
    memory_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    app_id: Optional[UUID] = None
    all_for_app: bool = False
    global_pause: bool = False
    state: Optional[MemoryState] = None
    user_id: str


@router.post("/actions/pause")
async def pause_memories(
    request: PauseMemoriesRequest,
    db: Session = Depends(get_db)
):
    
    global_pause = request.global_pause
    all_for_app = request.all_for_app
    app_id = request.app_id
    memory_ids = request.memory_ids
    category_ids = request.category_ids
    state = request.state or MemoryState.paused

    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user.id
    
    if global_pause:
        memories = db.query(Memory).filter(
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if app_id:
        memories = db.query(Memory).filter(
            Memory.app_id == app_id,
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused all memories for app {app_id}"}
    
    if all_for_app and memory_ids:
        memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.id.in_(memory_ids)
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if memory_ids:
        for memory_id in memory_ids:
            update_memory_state(db, memory_id, state, user_id)
        return {"message": f"Successfully paused {len(memory_ids)} memories"}

    if category_ids:
        memories = db.query(Memory).join(Memory.categories).filter(
            Category.id.in_(category_ids),
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused memories in {len(category_ids)} categories"}

    raise HTTPException(status_code=400, detail="Invalid pause request parameters")


@router.post("/actions/sync-storage")
async def sync_storage(
    user_id: str,
    direction: str = Query(..., description="'db-to-vector' or 'vector-to-db'"),
    db: Session = Depends(get_db)
):
    """
    Sync memories between SQLite database and Qdrant vector store.
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory client not available")
        
        qdrant_host = os.getenv('QDRANT_HOST', 'mem0_store')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = os.getenv('QDRANT_COLLECTION', 'openmemory')
        
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        db_memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted
        ).all()
        db_memory_ids = {str(mem.id) for mem in db_memories}
        
        try:
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True
            )
            vector_memory_ids = {str(point.id) for point, _ in scroll_result}
        except Exception as e:
            logging.error(f"Error reading from Qdrant: {e}")
            vector_memory_ids = set()
        
        if direction == "db-to-vector":
            orphaned_vectors = vector_memory_ids - db_memory_ids
            if orphaned_vectors:
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=list(orphaned_vectors)
                )
                return {
                    "message": f"Removed {len(orphaned_vectors)} orphaned vectors from Qdrant",
                    "removed_ids": list(orphaned_vectors)
                }
            return {"message": "No orphaned vectors found in Qdrant"}
            
        elif direction == "vector-to-db":
            orphaned_records = db_memory_ids - vector_memory_ids
            if orphaned_records:
                for memory_id in orphaned_records:
                    update_memory_state(db, UUID(memory_id), MemoryState.deleted, user.id)
                return {
                    "message": f"Removed {len(orphaned_records)} orphaned records from database",
                    "removed_ids": list(orphaned_records)
                }
            return {"message": "No orphaned records found in database"}
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid direction. Must be 'db-to-vector' or 'vector-to-db'"
            )
            
    except Exception as e:
        logging.error(f"Error syncing storage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync storage: {str(e)}")


@router.get("/{memory_id}/access-log")
async def get_memory_access_log(
    memory_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(MemoryAccessLog).filter(MemoryAccessLog.memory_id == memory_id)
    total = query.count()
    logs = query.order_by(MemoryAccessLog.accessed_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    for log in logs:
        app = db.query(App).filter(App.id == log.app_id).first()
        log.app_name = app.name if app else None

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "logs": logs
    }


class UpdateMemoryRequest(BaseModel):
    memory_content: str
    user_id: str


@router.put("/{memory_id}")
async def update_memory(
    memory_id: UUID,
    request: UpdateMemoryRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    memory = get_memory_or_404(db, memory_id)
    memory.content = request.memory_content
    db.commit()
    db.refresh(memory)
    return memory


class FilterMemoriesRequest(BaseModel):
    user_id: str
    page: int = 1
    size: int = 10
    search_query: Optional[str] = None
    app_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    sort_column: Optional[str] = None
    sort_direction: Optional[str] = None
    from_date: Optional[int] = None
    to_date: Optional[int] = None
    show_archived: Optional[bool] = False


@router.post("/filter", response_model=Page[MemoryResponse])
async def filter_memories(
    request: FilterMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
    )

    if not request.show_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    if request.search_query:
        query = query.filter(Memory.content.ilike(f"%{request.search_query}%"))

    if request.app_ids:
        query = query.filter(Memory.app_id.in_(request.app_ids))

    query = query.outerjoin(App, Memory.app_id == App.id)

    if request.category_ids:
        query = query.join(Memory.categories).filter(Category.id.in_(request.category_ids))
    else:
        query = query.outerjoin(Memory.categories)

    if request.from_date:
        from_datetime = datetime.fromtimestamp(request.from_date, tz=UTC)
        query = query.filter(Memory.created_at >= from_datetime)

    if request.to_date:
        to_datetime = datetime.fromtimestamp(request.to_date, tz=UTC)
        query = query.filter(Memory.created_at <= to_datetime)

    if request.sort_column and request.sort_direction:
        sort_direction = request.sort_direction.lower()
        if sort_direction not in ['asc', 'desc']:
            raise HTTPException(status_code=400, detail="Invalid sort direction")

        sort_mapping = {
            'memory': Memory.content,
            'app_name': App.name,
            'created_at': Memory.created_at
        }

        if request.sort_column not in sort_mapping:
            raise HTTPException(status_code=400, detail="Invalid sort column")

        sort_field = sort_mapping[request.sort_column]
        if sort_direction == 'desc':
            query = query.order_by(sort_field.desc())
        else:
            query = query.order_by(sort_field.asc())
    else:
        query = query.order_by(Memory.created_at.desc())

    query = query.options(
        joinedload(Memory.categories)
    ).distinct(Memory.id)

    return sqlalchemy_paginate(
        query,
        Params(page=request.page, size=request.size),
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
        ]
    )


# Enhanced related memories using BOTH categories AND graph
@router.get("/{memory_id}/related", response_model=Page[MemoryResponse])
async def get_related_memories(
    memory_id: UUID,
    user_id: str,
    use_graph: bool = Query(True, description="Use graph relationships for finding related memories"),
    params: Params = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    memory = get_memory_or_404(db, memory_id)
    
    # Get category-based related memories
    category_ids = [category.id for category in memory.categories]
    
    # Get graph-based related memories
    graph_memory_ids = []
    if use_graph:
        graph_memory_ids = get_related_memories_from_graph(memory_id, user_id, limit=20)
    
    # Combine both approaches
    if category_ids or graph_memory_ids:
        query = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.id != memory_id,
            Memory.state != MemoryState.deleted
        )
        
        # If we have graph relationships, prioritize those
        if graph_memory_ids:
            # Convert string IDs to UUID
            graph_uuids = [UUID(mid) for mid in graph_memory_ids]
            query = query.filter(Memory.id.in_(graph_uuids))
        elif category_ids:
            # Fall back to category-based search
            query = query.join(Memory.categories).filter(
                Category.id.in_(category_ids)
            )
        
        query = query.options(
            joinedload(Memory.categories),
            joinedload(Memory.app)
        ).order_by(Memory.created_at.desc())
    else:
        # No relationships found
        return Page.create([], total=0, params=params)
    
    params = Params(page=params.page, size=5)
    
    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=memory.metadata_
            )
            for memory in items
        ]
    )


# ========== GRAPH ENDPOINTS ==========

@router.get("/graph/stats")
async def get_graph_stats(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get statistics about the knowledge graph"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        memory_client = get_memory_client()
        if not memory_client or not hasattr(memory_client, 'graph'):
            return {
                "error": "Graph store not available",
                "graph_enabled": False
            }
        
        # Count total nodes
        node_result = query_graph(
            user_id,
            "MATCH (n) WHERE n.user_id = $user_id RETURN count(n) as count"
        )
        node_count = node_result[0]['count'] if node_result else 0
        
        # Count total relationships
        rel_result = query_graph(
            user_id,
            "MATCH (n)-[r]-(m) WHERE n.user_id = $user_id RETURN count(r) as count"
        )
        rel_count = rel_result[0]['count'] if rel_result else 0
        
        # Get entity types
        entity_result = query_graph(
            user_id,
            """
            MATCH (n) 
            WHERE n.user_id = $user_id 
            RETURN labels(n) as labels, count(*) as count
            ORDER BY count DESC
            LIMIT 10
            """
        )
        entity_types = [
            {"type": record["labels"][0] if record.get("labels") else "Unknown", 
             "count": record["count"]}
            for record in entity_result
        ]
        
        return {
            "graph_enabled": True,
            "total_entities": node_count,
            "total_relationships": rel_count,
            "entity_types": entity_types,
            "user_id": user_id
        }
        
    except Exception as e:
        logging.error(f"Error getting graph stats: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "graph_enabled": False
        }


@router.get("/{memory_id}/entities")
async def get_memory_graph_entities(
    memory_id: UUID,
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get entities extracted from a specific memory"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    memory = get_memory_or_404(db, memory_id)
    
    entities = get_memory_entities_from_graph(memory_id, user_id)
    
    return {
        "memory_id": str(memory_id),
        "entities": [
            {
                "name": e["name"],
                "type": e["types"][0] if e["types"] else "Unknown",
                "properties": e.get("properties", {})
            }
            for e in entities
        ],
        "count": len(entities)
    }


@router.get("/graph/search")
async def graph_search(
    query: str,
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Search for entities in the knowledge graph"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    cypher_query = """
    MATCH (n)
    WHERE n.user_id = $user_id 
    AND toLower(n.name) CONTAINS toLower($query)
    RETURN n.name as name, 
           labels(n) as types, 
           properties(n) as properties
    LIMIT $limit
    """
    
    results = query_graph(user_id, cypher_query, {"query": query, "limit": limit})
    
    return {
        "query": query,
        "results": [
            {
                "name": r["name"],
                "type": r["types"][0] if r["types"] else "Unknown",
                "properties": r.get("properties", {})
            }
            for r in results
        ],
        "count": len(results)
    }


@router.get("/graph/entity/{entity_name}/related")
async def get_related_entities(
    entity_name: str,
    user_id: str,
    depth: int = Query(1, ge=1, le=3),
    db: Session = Depends(get_db)
):
    """Get entities related to a specific entity"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    cypher_query = f"""
    MATCH path = (start)-[*1..{depth}]-(related)
    WHERE start.name = $entity_name 
    AND start.user_id = $user_id
    AND related.user_id = $user_id
    RETURN 
        start.name as source,
        related.name as target,
        type(relationships(path)[0]) as relationship,
        length(path) as distance
    LIMIT 50
    """
    
    results = query_graph(user_id, cypher_query, {"entity_name": entity_name})
    
    return {
        "entity": entity_name,
        "relationships": [
            {
                "source": r["source"],
                "target": r["target"],
                "relationship": r.get("relationship"),
                "distance": r["distance"]
            }
            for r in results
        ],
        "count": len(results)
    }


@router.get("/graph/entity/{entity_name}/memories")
async def get_memories_by_entity(
    entity_name: str,
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Find all memories that mention a specific entity"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    cypher_query = """
    MATCH (entity {name: $entity_name})-[:MENTIONED_IN]->(m:Memory)
    WHERE m.user_id = $user_id
    RETURN DISTINCT m.id as memory_id
    LIMIT $limit
    """
    
    results = query_graph(user_id, cypher_query, {"entity_name": entity_name, "limit": limit})
    memory_ids = [UUID(r["memory_id"]) for r in results if r.get("memory_id")]
    
    if not memory_ids:
        return {"entity": entity_name, "memories": [], "count": 0}
    
    # Get full memory details from database
    memories = db.query(Memory).filter(
        Memory.id.in_(memory_ids),
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted
    ).options(
        joinedload(Memory.categories),
        joinedload(Memory.app)
    ).all()
    
    return {
        "entity": entity_name,
        "memories": [
            {
                "id": str(memory.id),
                "content": memory.content,
                "created_at": memory.created_at.isoformat(),
                "app_name": memory.app.name if memory.app else None,
                "categories": [cat.name for cat in memory.categories]
            }
            for memory in memories
        ],
        "count": len(memories)
    }


@router.get("/graph/visualize")
async def get_graph_visualization(
    user_id: str,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Get graph data in format suitable for visualization"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        memory_client = get_memory_client()
        if not memory_client or not hasattr(memory_client, 'graph'):
            return {
                "error": "Graph store not available",
                "nodes": [],
                "edges": []
            }
        
        # Get nodes
        node_result = query_graph(
            user_id,
            """
            MATCH (n)
            WHERE n.user_id = $user_id
            RETURN id(n) as id, n.name as name, labels(n) as types
            LIMIT $limit
            """,
            {"limit": limit}
        )
        
        nodes = [
            {
                "id": record["id"],
                "label": record.get("name", "Unknown"),
                "type": record["types"][0] if record.get("types") else "Unknown"
            }
            for record in node_result
        ]
        
        # Get relationships
        edge_result = query_graph(
            user_id,
            """
            MATCH (a)-[r]->(b)
            WHERE a.user_id = $user_id
            RETURN id(a) as source, id(b) as target, type(r) as relationship
            LIMIT $limit
            """,
            {"limit": limit}
        )
        
        edges = [
            {
                "source": record["source"],
                "target": record["target"],
                "label": record.get("relationship", "RELATED_TO")
            }
            for record in edge_result
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
        
    except Exception as e:
        logging.error(f"Error getting graph visualization: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "nodes": [],
            "edges": []
        }


@router.get("/graph/context/{topic}")
async def get_topic_context(
    topic: str,
    user_id: str,
    depth: int = Query(2, ge=1, le=3, description="How many relationship hops to traverse"),
    include_memories: bool = Query(True, description="Include related memories"),
    db: Session = Depends(get_db)
):
    """Get comprehensive context about a topic by traversing the knowledge graph.
    Perfect for AI coding agents to understand full context about a system/feature."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Find matching entities
    cypher_query = f"""
    MATCH (start)
    WHERE start.user_id = $user_id 
    AND toLower(start.name) CONTAINS toLower($topic)
    WITH start
    MATCH path = (start)-[*0..{depth}]-(related)
    WHERE related.user_id = $user_id
    RETURN DISTINCT
        start.name as root_entity,
        related.name as related_entity,
        labels(related) as entity_type,
        length(path) as distance
    ORDER BY distance, related_entity
    LIMIT 100
    """
    
    results = query_graph(user_id, cypher_query, {"topic": topic})
    
    # Organize entities by distance
    context = {
        "topic": topic,
        "root_entities": list(set([r["root_entity"] for r in results])),
        "related_entities": {},
        "memories": []
    }
    
    for r in results:
        distance = r["distance"]
        if distance not in context["related_entities"]:
            context["related_entities"][distance] = []
        context["related_entities"][distance].append({
            "name": r["related_entity"],
            "type": r["entity_type"][0] if r["entity_type"] else "Unknown"
        })
    
    # Get related memories if requested
    if include_memories and context["root_entities"]:
        for entity_name in context["root_entities"][:5]:  # Limit to first 5 root entities
            entity_memories = await get_memories_by_entity(entity_name, user_id, limit=10, db=db)
            context["memories"].extend(entity_memories["memories"])
    
    return context