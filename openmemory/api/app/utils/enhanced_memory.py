"""
Enhanced memory utilities for intelligent memory management.
Provides hybrid search, smart memory addition, and relationship analysis.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from app.database import SessionLocal
from app.models import Memory, MemoryState
from app.utils.memory import get_memory_client


@dataclass
class MemorySearchResult:
    """Enhanced search result combining vector, graph, and temporal data"""
    id: str
    content: str
    score: float
    source: str  # 'vector', 'graph', 'temporal'
    metadata: Dict[str, Any]
    relationships: List[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class MemoryAdditionResult:
    """Result of smart memory addition"""
    added_memories: List[Dict[str, Any]]
    updated_memories: List[Dict[str, Any]]
    skipped_facts: List[str]
    related_memories: List[Dict[str, Any]]
    status: str  # 'new', 'updated', 'merged', 'duplicate'
    summary: str


class EnhancedMemoryManager:
    """Enhanced memory manager with hybrid search and smart addition"""
    
    def __init__(self):
        self.memory_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize memory client safely"""
        try:
            self.memory_client = get_memory_client()
        except Exception as e:
            logging.warning(f"Failed to initialize memory client: {e}")
            self.memory_client = None
    
    def hybrid_search(self, query: str, user_id: str, limit: int = 10) -> List[MemorySearchResult]:
        """
        Perform hybrid search across vector, graph, and temporal dimensions
        """
        if not self.memory_client:
            return []
        
        results = []
        
        # 1. Vector similarity search
        try:
            vector_results = self._vector_search(query, user_id, limit)
            results.extend(vector_results)
        except Exception as e:
            logging.warning(f"Vector search failed: {e}")
        
        # 2. Graph relationship search
        try:
            graph_results = self._graph_search(query, user_id, limit)
            results.extend(graph_results)
        except Exception as e:
            logging.warning(f"Graph search failed: {e}")
        
        # 3. Temporal context search
        try:
            temporal_results = self._temporal_search(query, user_id, limit)
            results.extend(temporal_results)
        except Exception as e:
            logging.warning(f"Temporal search failed: {e}")
        
        # 4. Deduplicate and rank results
        return self._deduplicate_and_rank(results, limit)
    
    def _vector_search(self, query: str, user_id: str, limit: int) -> List[MemorySearchResult]:
        """Vector similarity search in Qdrant"""
        results = []
        
        try:
            filters = {"user_id": user_id}
            embeddings = self.memory_client.embedding_model.embed(query, "search")
            
            hits = self.memory_client.vector_store.search(
                query=query,
                vectors=embeddings,
                limit=limit,
                filters=filters,
            )
            
            for hit in hits:
                results.append(MemorySearchResult(
                    id=hit.id,
                    content=hit.payload.get("data", ""),
                    score=hit.score,
                    source="vector",
                    metadata=hit.payload,
                    created_at=hit.payload.get("created_at"),
                    updated_at=hit.payload.get("updated_at")
                ))
        except Exception as e:
            logging.warning(f"Vector search error: {e}")
        
        return results
    
    def _graph_search(self, query: str, user_id: str, limit: int) -> List[MemorySearchResult]:
        """Graph relationship search in Neo4j"""
        results = []
        
        try:
            if hasattr(self.memory_client, 'graph') and self.memory_client.graph:
                # Extract entities from query
                entities = self._extract_entities_from_query(query)
                
                # Search for related entities and relationships
                for entity in entities:
                    related = self._find_related_entities(entity, user_id)
                    for rel in related:
                        results.append(MemorySearchResult(
                            id=rel.get('id', str(uuid.uuid4())),
                            content=rel.get('content', ''),
                            score=rel.get('relevance', 0.7),
                            source="graph",
                            metadata=rel,
                            relationships=rel.get('relationships', [])
                        ))
        except Exception as e:
            logging.warning(f"Graph search error: {e}")
        
        return results
    
    def _temporal_search(self, query: str, user_id: str, limit: int) -> List[MemorySearchResult]:
        """Temporal context search based on recency and patterns"""
        results = []
        
        try:
            from app.utils.db import get_user_and_app
            db = SessionLocal()
            try:
                # Get user object to get the UUID
                try:
                    user, _ = get_user_and_app(db, user_id=user_id, app_id="temporal_search")
                    actual_user_id = user.id
                except Exception:
                    # If we can't get user, skip temporal search
                    return results
                
                # Get recent memories using the actual UUID
                current_time = datetime.utcnow()
                recent_memories = db.query(Memory).filter(
                    Memory.user_id == actual_user_id,
                    Memory.state == MemoryState.active,
                    Memory.created_at >= current_time - timedelta(days=30)
                ).order_by(Memory.created_at.desc()).limit(limit).all()
                
                # Score based on recency and relevance
                for memory in recent_memories:
                    age_days = (current_time - memory.created_at).days
                    recency_score = max(0.3, 1.0 - (age_days / 30))  # Decay over 30 days
                    
                    # Simple relevance check
                    relevance = self._calculate_text_relevance(query, memory.content)
                    combined_score = (recency_score * 0.3) + (relevance * 0.7)
                    
                    if combined_score > 0.4:  # Threshold for inclusion
                        results.append(MemorySearchResult(
                            id=str(memory.id),
                            content=memory.content,
                            score=combined_score,
                            source="temporal",
                            metadata={"age_days": age_days, "recency_score": recency_score},
                            created_at=memory.created_at,
                            updated_at=memory.updated_at
                        ))
            finally:
                db.close()
        except Exception as e:
            logging.warning(f"Temporal search error: {e}")
        
        return results
    
    def smart_add_memory(self, text: str, user_id: str, metadata: Dict = None) -> MemoryAdditionResult:
        """
        Intelligently add memory by checking existing memories and extracting only new facts
        """
        if not self.memory_client:
            return MemoryAdditionResult([], [], [], [], "error", "Memory system unavailable")
        
        # 1. Search for existing related memories
        existing_memories = self.hybrid_search(text, user_id, limit=20)
        
        # 2. Extract facts from new text
        new_facts = self._extract_facts(text)
        
        # 3. Compare with existing facts
        existing_facts = []
        for memory in existing_memories:
            if memory.score > 0.7:  # High similarity threshold
                facts = self._extract_facts(memory.content)
                existing_facts.extend(facts)
        
        # 4. Identify truly new facts
        novel_facts = self._find_novel_facts(new_facts, existing_facts)
        
        # 5. Add only new information
        if novel_facts:
            novel_text = ". ".join(novel_facts)
            response = self.memory_client.add(
                novel_text,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            status = "new" if not existing_memories else "updated"
            summary = f"Added {len(novel_facts)} new facts. Found {len(existing_facts)} existing related facts."
            
            return MemoryAdditionResult(
                added_memories=response.get('results', []) if response else [],
                updated_memories=[],
                skipped_facts=[f for f in new_facts if f not in novel_facts],
                related_memories=[m.__dict__ for m in existing_memories[:5]],
                status=status,
                summary=summary
            )
        else:
            return MemoryAdditionResult(
                added_memories=[],
                updated_memories=[],
                skipped_facts=new_facts,
                related_memories=[m.__dict__ for m in existing_memories[:5]],
                status="duplicate",
                summary=f"No new facts found. All {len(new_facts)} facts already exist in memory."
            )
    
    def comprehensive_memory_handle(self, user_message: str, llm_response: str, user_id: str) -> Dict[str, Any]:
        """
        Comprehensive memory processing like human memory - extract, store, relate, and provide context
        """
        result = {
            "extracted_memories": [],
            "related_context": [],
            "memory_updates": [],
            "insights": [],
            "status": "processed"
        }
        
        try:
            # 1. Extract memorable information from both messages
            memorable_content = self._extract_memorable_content(user_message, llm_response)
            
            # 2. Search for related existing memories
            related_memories = []
            for content in memorable_content:
                related = self.hybrid_search(content, user_id, limit=10)
                related_memories.extend(related)
            
            # 3. Smart addition of new memories
            for content in memorable_content:
                addition_result = self.smart_add_memory(content, user_id)
                result["memory_updates"].append({
                    "content": content,
                    "result": addition_result.__dict__
                })
            
            # 4. Provide contextual insights
            result["related_context"] = [m.__dict__ for m in related_memories[:10]]
            result["insights"] = self._generate_memory_insights(related_memories, memorable_content)
            
            # 5. Identify patterns and connections
            result["patterns"] = self._identify_conversation_patterns(user_message, llm_response, related_memories)
            
        except Exception as e:
            logging.error(f"Comprehensive memory handle error: {e}")
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entities from search query"""
        # Simple implementation - could be enhanced with NER
        words = query.lower().split()
        # Filter out common words, keep potential entities
        entities = [w for w in words if len(w) > 2 and w not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'who']]
        return entities[:5]  # Limit to avoid too many queries
    
    def _find_related_entities(self, entity: str, user_id: str) -> List[Dict]:
        """Find entities related to given entity in graph"""
        # This would query Neo4j for relationships
        # Simplified implementation
        return []
    
    def _calculate_text_relevance(self, query: str, text: str) -> float:
        """Simple text relevance calculation"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words)
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text"""
        # Simple sentence splitting - could be enhanced with NLP
        import re
        sentences = re.split(r'[.!?]+', text)
        facts = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return facts
    
    def _find_novel_facts(self, new_facts: List[str], existing_facts: List[str]) -> List[str]:
        """Identify facts that are truly new"""
        novel = []
        for new_fact in new_facts:
            is_novel = True
            for existing_fact in existing_facts:
                similarity = self._calculate_text_relevance(new_fact, existing_fact)
                if similarity > 0.8:  # High similarity threshold
                    is_novel = False
                    break
            if is_novel:
                novel.append(new_fact)
        return novel
    
    def _extract_memorable_content(self, user_message: str, llm_response: str) -> List[str]:
        """Extract content worth remembering from conversation"""
        memorable = []
        
        # Extract from user message (preferences, facts, context)
        user_facts = self._extract_facts(user_message)
        memorable.extend(user_facts)
        
        # Extract from LLM response (important information provided)
        llm_facts = self._extract_facts(llm_response)
        # Filter LLM facts to focus on substantive information
        substantive_llm_facts = [f for f in llm_facts if len(f) > 20 and any(keyword in f.lower() for keyword in ['recommend', 'suggest', 'important', 'remember', 'note', 'consider'])]
        memorable.extend(substantive_llm_facts)
        
        return memorable
    
    def _generate_memory_insights(self, related_memories: List[MemorySearchResult], new_content: List[str]) -> List[str]:
        """Generate insights from memory patterns"""
        insights = []
        
        if len(related_memories) > 5:
            insights.append(f"Found {len(related_memories)} related memories - this appears to be an ongoing topic of interest")
        
        # Analyze sources
        sources = [m.source for m in related_memories]
        if sources.count('temporal') > sources.count('vector'):
            insights.append("Recent conversations show continued interest in this topic")
        
        if sources.count('graph') > 0:
            insights.append("This topic has established relationships with other concepts in memory")
        
        return insights
    
    def _identify_conversation_patterns(self, user_message: str, llm_response: str, related_memories: List[MemorySearchResult]) -> List[str]:
        """Identify patterns in conversation flow"""
        patterns = []
        
        if len(related_memories) > 0:
            patterns.append("Building on previous conversations")
        
        if any(word in user_message.lower() for word in ['remember', 'recall', 'before', 'previous']):
            patterns.append("User referencing past information")
        
        if any(word in llm_response.lower() for word in ['as we discussed', 'previously', 'remember']):
            patterns.append("LLM using memory context")
        
        return patterns
    
    def _deduplicate_and_rank(self, results: List[MemorySearchResult], limit: int) -> List[MemorySearchResult]:
        """Remove duplicates and rank by combined relevance"""
        # Simple deduplication by ID
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        # Sort by score (descending)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results[:limit]


# Global instance
enhanced_memory_manager = EnhancedMemoryManager()