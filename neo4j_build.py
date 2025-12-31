import json
import os
import logging
import time
from typing import List

from neo4j import GraphDatabase
import ollama
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database and Model Settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
OLLAMA_API_HOST = os.getenv("OLLAMA_API_HOST", "http://localhost:11434")

DATA_DIR = "data/"
BATCH_SIZE = 8


class GraphArchitect:
    """Builds Neo4j graph from JSONL data files with vector embeddings."""

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.client = ollama.Client(host=OLLAMA_API_HOST)
        logger.info("Neo4j connection and Ollama client ready.")

    def close(self):
        """Close database connection."""
        self.driver.close()

    def setup_schema(self):
        """Create Neo4j schema constraints and indexes."""
        logger.info("Creating database schema and indexes")
        
        queries = [
            "CREATE CONSTRAINT asset_id_unique IF NOT EXISTS FOR (a:Asset) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT macro_id_unique IF NOT EXISTS FOR (m:MacroVariable) REQUIRE m.id IS UNIQUE",
            "CREATE VECTOR INDEX event_embedding_index IF NOT EXISTS FOR (e:Event) ON (e.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}",
            "CREATE INDEX event_date_index IF NOT EXISTS FOR (e:Event) ON (e.date)",
            "CREATE INDEX asset_sector_index IF NOT EXISTS FOR (a:Asset) ON (a.sector)"
        ]
        
        with self.driver.session() as session:
            for query in queries:
                session.run(query)
        
        logger.info("Schema setup completed.")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings(model=EMBEDDING_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            logger.error("Embedding error: %s", e)
            return []

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        return [self._generate_embedding(text) for text in texts]

    def _load_jsonl(self, file_path: str) -> List[dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _ingest_nodes(self, file_path: str, label: str, batch_size: int):
        """Ingest nodes of a specific label from JSONL file."""
        data = self._load_jsonl(file_path)
        
        query = f"""
        UNWIND $batch AS row
        MERGE (n:{label} {{id: row.id}})
        SET n += row
        """

        with self.driver.session() as session:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                session.run(query, batch=batch)

    def _ingest_edges(self, session, file_path: str, cypher_query: str, batch_size: int = 1000):
        """Ingest edges from JSONL file using provided Cypher query."""
        data = self._load_jsonl(file_path)
        
        for i in tqdm(range(0, len(data), batch_size), desc="Loading Edges"):
            batch = data[i:i + batch_size]
            session.run(cypher_query, batch=batch)

    def ingest_assets(self):
        """Load Asset nodes into the graph."""
        file_path = os.path.join(DATA_DIR, "nodes_assets.jsonl")
        
        if not os.path.exists(file_path):
            logger.warning("Asset file not found, skipping.")
            return

        logger.info("Loading Asset nodes")
        self._ingest_nodes(file_path, "Asset", batch_size=BATCH_SIZE)

    def ingest_macro_variables(self):
        """Load MacroVariable nodes into the graph."""
        file_path = os.path.join(DATA_DIR, "nodes_macro.jsonl")
        
        if not os.path.exists(file_path):
            return

        logger.info("Loading MacroVariable nodes")
        self._ingest_nodes(file_path, "MacroVariable", batch_size=BATCH_SIZE)

    def _process_event_batch(self, batch: List[dict]) -> List[dict]:
        """Process event batch: generate embeddings and filter valid items."""
        texts = [f"{item['headline']}\n{item['body']}" for item in batch]
        embeddings = self.generate_embeddings_batch(texts)
        
        processed = []
        for item, vector in zip(batch, embeddings):
            if vector:
                item['embedding'] = vector
                processed.append(item)
        
        return processed

    def ingest_events_with_embeddings(self):
        """Load Event nodes with vector embeddings."""
        file_path = os.path.join(DATA_DIR, "nodes_events.jsonl")
        
        if not os.path.exists(file_path):
            return

        logger.info("Loading Event nodes with '%s' embeddings", EMBEDDING_MODEL)
        
        data = self._load_jsonl(file_path)
        
        cypher_query = """
        UNWIND $batch AS row
        MERGE (e:Event {id: row.id})
        SET e.headline = row.headline,
            e.body = row.body,
            e.type = row.type,
            e.date = date(row.date),
            e.affected_sector = row.affected_sector,
            e.regime = row.regime,
            e.embedding = row.embedding
        """
        
        with self.driver.session() as session:
            for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Event Embedding & Ingestion"):
                batch = data[i:i + BATCH_SIZE]
                processed_batch = self._process_event_batch(batch)
                session.run(cypher_query, batch=processed_batch)

    def _get_edge_configs(self) -> List[tuple]:
        """Return edge file configurations (filename, relation_type, cypher_query)."""
        return [
            (
                "edges_event_asset.jsonl",
                "AFFECTS",
                "UNWIND $batch AS row "
                "MATCH (s:Event {id: row.source}) "
                "MATCH (t:Asset {id: row.target}) "
                "MERGE (s)-[r:AFFECTS]->(t) "
                "SET r.weight = row.weight, r.causal_lag = row.causal_lag, r.is_ground_truth = row.is_ground_truth"
            ),
            (
                "edges_asset_correlation.jsonl",
                "CO_MOVES_WITH",
                "UNWIND $batch AS row "
                "MATCH (s:Asset {id: row.source}) "
                "MATCH (t:Asset {id: row.target}) "
                "MERGE (s)-[r:CO_MOVES_WITH]->(t) "
                "SET r.correlation = row.correlation, r.direction = row.direction"
            ),
            (
                "edges_asset_macro.jsonl",
                "SENSITIVE_TO",
                "UNWIND $batch AS row "
                "MATCH (s:Asset {id: row.source}) "
                "MATCH (t:MacroVariable {id: row.target}) "
                "MERGE (s)-[r:SENSITIVE_TO]->(t) "
                "SET r.beta = row.beta_coefficient, r.factor = row.factor"
            )
        ]

    def ingest_all_edges(self):
        """Load all relationship edges into the graph."""
        edge_configs = self._get_edge_configs()

        with self.driver.session() as session:
            for filename, rel_type, cypher in edge_configs:
                file_path = os.path.join(DATA_DIR, filename)
                
                if not os.path.exists(file_path):
                    continue
                
                logger.info("Loading relationships: %s (%s)", filename, rel_type)
                self._ingest_edges(session, file_path, cypher)

    def run_pipeline(self):
        """Execute the complete graph building pipeline."""
        start_time = time.time()
        
        # Schema Setup
        self.setup_schema()
        
        # Node Ingestion
        self.ingest_assets()
        self.ingest_macro_variables()
        self.ingest_events_with_embeddings()
        
        # Edge Ingestion
        self.ingest_all_edges()
        
        elapsed = time.time() - start_time
        logger.info("Completed! Total time: %.2f seconds.", elapsed)


if __name__ == "__main__":
    architect = GraphArchitect()
    try:
        architect.run_pipeline()
    finally:
        architect.close()