"""
Enhanced memory utilities for intelligent memory management.
Provides hybrid search, smart memory addition, and relationship analysis.
"""

import json
import logging
import re
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
        novel_facts = self._find_novel_facts(new_facts, existing_facts, user_id=user_id)
        
        # 5. Add only new information
        #    - Store each fact individually in vector store (infer=False, graph=False)
        #    - Do ONE batched graph extraction with the full original text
        if novel_facts:
            all_results = []
            for fact in novel_facts:
                try:
                    # infer=False: skip LLM re-extraction
                    # graph=False: skip graph extraction per-fact (we batch it below)
                    response = self.memory_client.add(
                        [{"role": "user", "content": fact}],
                        user_id=user_id,
                        metadata=metadata or {},
                        infer=False,
                        graph=False,
                    )
                    all_results.extend(response.get('results', []))
                except Exception as e:
                    logging.warning(f"Failed to add fact '{fact[:50]}': {e}")

            # 6. Single graph extraction call with the FULL original text
            #    This gives the graph LLM enough context to extract meaningful
            #    entities and relationships (not fragments)
            try:
                if hasattr(self.memory_client, 'enable_graph') and self.memory_client.enable_graph:
                    graph_result = self.memory_client._add_to_graph(
                        [{"role": "user", "content": text}],
                        {"user_id": user_id},
                    )
                    logging.info(f"Graph extraction from full text: {graph_result}")
            except Exception as e:
                logging.warning(f"Graph extraction failed (non-fatal): {e}")

            response = {'results': all_results}

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
        import re as _re
        # Strip punctuation, lowercase, split
        cleaned = _re.sub(r'[^\w\s]', '', query.lower())
        words = cleaned.split()
        # Filter out common/stop words, keep potential entities
        stopwords = frozenset([
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'way', 'what', 'does', 'this', 'that', 'with',
            'from', 'have', 'been', 'will', 'would', 'could', 'should', 'there',
            'their', 'them', 'they', 'then', 'than', 'when', 'where', 'which',
            'about', 'into', 'some', 'also', 'just', 'more', 'very', 'much',
            'work', 'used', 'main', 'exist',
        ])
        entities = [w for w in words if len(w) > 2 and w not in stopwords]
        return entities[:5]
    
    def _find_related_entities(self, entity: str, user_id: str) -> List[Dict]:
        """Find entities related to given entity in Neo4j graph"""
        results = []
        try:
            if not (hasattr(self.memory_client, 'graph') and self.memory_client.graph):
                return results
            graph = self.memory_client.graph
            if not hasattr(graph, 'graph'):
                return results
            driver = graph.graph.driver if hasattr(graph.graph, 'driver') else None
            if not driver:
                # Try _driver attribute (neo4j graph store)
                driver = getattr(graph.graph, '_driver', None)
            if not driver:
                return results

            query = """
            MATCH (n)-[r]->(m)
            WHERE (toLower(n.name) CONTAINS toLower($entity)
                   OR toLower(m.name) CONTAINS toLower($entity))
            AND n.name <> m.name
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT 5
            """
            with driver.session() as session:
                records = session.run(query, entity=entity)
                for record in records:
                    src = record['source'].replace('_', ' ')
                    rel = record['relationship'].replace('_', ' ')
                    tgt = record['target'].replace('_', ' ')
                    content = f"{src} {rel} {tgt}"
                    results.append({
                        'content': content,
                        'relevance': 0.65,
                        'relationships': [
                            {'source': record['source'], 'rel': record['relationship'], 'target': record['target']}
                        ]
                    })
        except Exception as e:
            logging.warning(f"Graph entity search failed for '{entity}': {e}")
        return results
    
    def _calculate_text_relevance(self, query: str, text: str) -> float:
        """Simple text relevance calculation"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words)
    
    # File extensions and version-like patterns that should NOT be treated as sentence endings
    _EXTENSION_RE = re.compile(
        r'\.'
        r'(?:php|js|jsx|ts|tsx|py|json|yaml|yml|toml|xml|html|css|scss|md|txt|sh|bat|'
        r'env|lock|sql|csv|log|conf|cfg|ini|vue|svelte|rb|rs|go|java|kt|swift|c|cpp|h|'
        r'hpp|cs|fs|ex|exs|erl|hs|ml|r|jl|dart|lua|pl|pm|ipynb)'
        r'(?=[\s,;:)\]}\-/"\']|$)',
        re.IGNORECASE
    )
    _VERSION_RE = re.compile(r'(?:v?\d+)\.\d+(?:\.\d+)*(?:\+|-\w+)*')
    _PATH_RE = re.compile(r'[\w/\\]+\.[\w]+')
    _PLACEHOLDER = '\x00DOT\x00'

    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text, protecting file extensions and versions."""
        # 1. Protect file extensions, version numbers, and paths from sentence splitting
        protected = text
        replacements = []

        # Protect version numbers first (e.g., "v3.3", "PHP 8.2+", "1.17.0")
        for m in self._VERSION_RE.finditer(protected):
            replacements.append(m.group())
        for r in replacements:
            protected = protected.replace(r, r.replace('.', self._PLACEHOLDER))

        # Protect file extensions (e.g., ".php", ".js")
        protected = self._EXTENSION_RE.sub(
            lambda m: m.group().replace('.', self._PLACEHOLDER), protected
        )

        # 2. Split on actual sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', protected)

        # 3. Restore protected dots and validate
        facts = []
        for s in sentences:
            restored = s.replace(self._PLACEHOLDER, '.').strip()
            # Strip trailing punctuation for cleaner facts
            restored = restored.rstrip('.!? ')
            if not restored or len(restored) < 20:
                continue
            # Reject fragments starting with orphaned extensions/lowercase
            if re.match(r'^[a-z]{1,4}[)\]\s,]', restored):
                continue
            facts.append(restored)

        return facts
    
    def _find_novel_facts(self, new_facts: List[str], existing_facts: List[str], user_id: str = None) -> List[str]:
        """Identify facts that are truly new using Qdrant vector similarity"""
        if not self.memory_client:
            return new_facts

        novel = []
        for new_fact in new_facts:
            try:
                # Check directly against Qdrant for semantic duplicates
                emb = self.memory_client.embedding_model.embed(new_fact, "search")
                filters = {"user_id": user_id} if user_id else {}
                hits = self.memory_client.vector_store.search(
                    query=new_fact, vectors=emb, limit=1, filters=filters
                )
                best_score = hits[0].score if hits else 0
                if best_score >= 0.85:
                    logging.info(f"Dedup (enhanced): skip '{new_fact[:60]}' (score={best_score:.3f})")
                    continue
            except Exception as e:
                logging.warning(f"Vector dedup failed, keeping fact: {e}")
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
        """Remove duplicates, interleave sources, and rank by relevance."""
        # Deduplication by ID and content
        seen_ids = set()
        seen_content = set()
        unique_results = []

        for result in results:
            content_key = result.content.lower().strip()[:60]
            if result.id not in seen_ids and content_key not in seen_content:
                seen_ids.add(result.id)
                seen_content.add(content_key)
                unique_results.append(result)

        # Sort by score within each source
        vector_results = sorted([r for r in unique_results if r.source == 'vector'],
                                key=lambda x: x.score, reverse=True)
        graph_results = sorted([r for r in unique_results if r.source == 'graph'],
                               key=lambda x: x.score, reverse=True)
        temporal_results = sorted([r for r in unique_results if r.source == 'temporal'],
                                  key=lambda x: x.score, reverse=True)

        # Interleave: prioritize vector (most informative), then graph, then temporal
        # Take up to 60% vector, 30% graph, 10% temporal
        max_vector = max(1, int(limit * 0.6))
        max_graph = max(1, int(limit * 0.3))
        max_temporal = max(1, int(limit * 0.1))

        merged = []
        merged.extend(vector_results[:max_vector])
        merged.extend(graph_results[:max_graph])
        merged.extend(temporal_results[:max_temporal])

        # If we haven't hit the limit, fill with remaining results
        remaining = [r for r in unique_results if r not in merged]
        remaining.sort(key=lambda x: x.score, reverse=True)
        merged.extend(remaining[:limit - len(merged)])

        # Final sort by score
        merged.sort(key=lambda x: x.score, reverse=True)

        return merged[:limit]


# Global instance
enhanced_memory_manager = EnhancedMemoryManager()