from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import ollama

from agents.tools.base_tool import BaseTool
from agents.models.schemas import (
    SemanticSearchInput, 
    CypherQueryInput,
    SemanticSearchResult,
    CypherQueryResult
)
from agents.config import NEO4J_URI, NEO4J_AUTH, EMBEDDING_MODEL


class SemanticSearchTool(BaseTool):
    """
    Semantic search using vector similarity on Event nodes
    """
    
    def __init__(self):
        super().__init__()
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.embedding_client = ollama.Client()
        
    @property
    def name(self) -> str:
        return "semantic_search"
    
    @property
    def description(self) -> str:
        return """Search financial events by semantic similarity. 
        Use this for text-based queries like 'find events about interest rate changes' 
        or 'what happened during the crisis in 2026'."""
    
    @property
    def input_schema(self) -> type[SemanticSearchInput]:
        return SemanticSearchInput
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text"""
        response = self.embedding_client.embeddings(
            model=EMBEDDING_MODEL, 
            prompt=text
        )
        return response['embedding']
    
    def _execute(
        self, 
        query: str, 
        limit: int = 5,
        filter_type: Optional[str] = None,
        filter_sector: Optional[str] = None,
        filter_regime: Optional[str] = None
    ) -> List[SemanticSearchResult]:
        """
        Execute semantic search
        
        Args:
            query: Search query text
            limit: Maximum results
            filter_type: Filter by event type
            filter_sector: Filter by affected sector
            filter_regime: Filter by regime
            
        Returns:
            List of SemanticSearchResult
        """
        # Generate query embedding
        query_vector = self._generate_embedding(query)
        
        # Build Cypher query with optional filters
        cypher = """
        CALL db.index.vector.queryNodes('event_embedding_index', $limit, $embedding)
        YIELD node AS e, score
        """
        
        # Add filters
        where_clauses = []
        if filter_type:
            where_clauses.append("e.type = $filter_type")
        if filter_sector:
            where_clauses.append("e.affected_sector = $filter_sector")
        if filter_regime:
            where_clauses.append("e.regime = $filter_regime")
        
        if where_clauses:
            cypher += "WHERE " + " AND ".join(where_clauses) + "\n"
        
        cypher += """
        RETURN e.id AS event_id,
               e.headline AS headline,
               e.body AS body,
               e.type AS event_type,
               e.date AS date,
               e.affected_sector AS affected_sector,
               e.regime AS regime,
               score AS similarity_score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        # Execute query with error handling
        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    embedding=query_vector,
                    limit=limit,
                    filter_type=filter_type,
                    filter_sector=filter_sector,
                    filter_regime=filter_regime
                )
                
                records = [
                    SemanticSearchResult(
                        event_id=record["event_id"],
                        headline=record["headline"],
                        body=record["body"],
                        event_type=record["event_type"],
                        date=record["date"],
                        affected_sector=record["affected_sector"],
                        regime=record["regime"],
                        similarity_score=record["similarity_score"]
                    )
                    for record in result
                ]
                
            return records
            
        except Exception as e:
            # Fallback to text-based search if vector index fails
            self.logger.warning(f"Vector search failed: {e}. Falling back to text search.")
            return self._fallback_text_search(query, limit, filter_type, filter_sector, filter_regime)
    
    def _fallback_text_search(
        self, 
        query: str, 
        limit: int,
        filter_type: Optional[str] = None,
        filter_sector: Optional[str] = None,
        filter_regime: Optional[str] = None
    ) -> List[SemanticSearchResult]:
        """Fallback text-based search when vector index is unavailable"""
        cypher = """
        MATCH (e:Event)
        WHERE toLower(e.headline) CONTAINS toLower($query) 
           OR toLower(e.body) CONTAINS toLower($query)
        """
        
        if filter_type:
            cypher += " AND e.type = $filter_type"
        if filter_sector:
            cypher += " AND e.affected_sector = $filter_sector"
        if filter_regime:
            cypher += " AND e.regime = $filter_regime"
        
        cypher += """
        RETURN e.id AS event_id,
               e.headline AS headline,
               e.body AS body,
               e.type AS event_type,
               e.date AS date,
               e.affected_sector AS affected_sector,
               e.regime AS regime,
               1.0 AS similarity_score
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(
                cypher,
                query=query,
                limit=limit,
                filter_type=filter_type,
                filter_sector=filter_sector,
                filter_regime=filter_regime
            )
            
            records = [
                SemanticSearchResult(
                    event_id=record["event_id"],
                    headline=record["headline"],
                    body=record["body"],
                    event_type=record["event_type"],
                    date=record["date"],
                    affected_sector=record["affected_sector"],
                    regime=record["regime"],
                    similarity_score=record["similarity_score"]
                )
                for record in result
            ]
        
        return records
    
    def __del__(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()


class CypherQueryTool(BaseTool):
    """
    Execute custom Cypher queries on Neo4j
    """
    
    def __init__(self):
        super().__init__()
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        
    @property
    def name(self) -> str:
        return "cypher_query"
    
    @property
    def description(self) -> str:
        return """Execute custom Cypher queries for complex graph analysis.
        Use this for relationship queries, aggregations, and graph traversals.
        
        Schema:
        - Nodes: Event (headline, body, type, date, affected_sector, regime)
                 Asset (id, sector)
                 MacroVariable (id, name)
        - Relationships: AFFECTS (Event->Asset, weight, causal_lag)
                        CO_MOVES_WITH (Asset->Asset, correlation, direction)
                        SENSITIVE_TO (Asset->MacroVariable, beta, factor)
        """
    
    @property
    def input_schema(self) -> type[CypherQueryInput]:
        return CypherQueryInput
    
    def _execute(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> CypherQueryResult:
        """
        Execute Cypher query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            CypherQueryResult with records and count
        """
        if parameters is None:
            parameters = {}
        
        with self.driver.session() as session:
            result = session.run(query, parameters)
            records = [record.data() for record in result]
            
        return CypherQueryResult(
            records=records,
            count=len(records)
        )
    
    def __del__(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()


def get_neo4j_schema() -> Dict[str, Any]:
    """
    Fetch Neo4j schema information for LLM context
    
    Returns:
        Dict with node labels, relationships, and properties
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    with driver.session() as session:
        # Get node labels
        labels_result = session.run("CALL db.labels()")
        node_labels = [record["label"] for record in labels_result]
        
        # Get relationship types
        rels_result = session.run("CALL db.relationshipTypes()")
        relationship_types = [record["relationshipType"] for record in rels_result]
        
        # Get node properties
        node_properties = {}
        for label in node_labels:
            props_result = session.run(
                f"MATCH (n:{label}) RETURN keys(n) AS props LIMIT 1"
            )
            props_record = props_result.single()
            if props_record:
                node_properties[label] = props_record["props"]
        
        # Get relationship properties
        relationship_properties = {}
        for rel_type in relationship_types:
            props_result = session.run(
                f"MATCH ()-[r:{rel_type}]->() RETURN keys(r) AS props LIMIT 1"
            )
            props_record = props_result.single()
            if props_record:
                relationship_properties[rel_type] = props_record["props"]
    
    driver.close()
    
    return {
        "node_labels": node_labels,
        "relationship_types": relationship_types,
        "node_properties": node_properties,
        "relationship_properties": relationship_properties
    }