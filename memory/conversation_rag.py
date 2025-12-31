import lancedb
import ollama
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding configuration
EMBEDDING_MODEL = "embeddinggemma:300m"
EMBEDDING_DIM = 768


class ConversationRAG:
    """
    RAG-based conversation memory using LanceDB
    
    Features:
    - Store messages with embeddings
    - Semantic search for relevant past conversations
    - Get context around matched messages
    """
    
    def __init__(self, db_path: str = "memory/lancedb_store"):
        """
        Initialize ConversationRAG
        
        Args:
            db_path: Path to LanceDB storage directory
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LanceDB
        self.db = lancedb.connect(str(self.db_path))
        
        # Initialize Ollama client for embeddings
        self.ollama_client = ollama.Client()
        
        # Create or get table
        self._init_table()
        
        logger.info(f"ConversationRAG initialized with db at {self.db_path}")
    
    def _init_table(self):
        """Initialize LanceDB table if not exists"""
        table_name = "conversations"
        
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)
        else:
            # Create table with initial schema
            initial_data = [{
                "id": "init",
                "conversation_id": "init",
                "role": "system",
                "content": "Conversation memory initialized",
                "timestamp": datetime.now().isoformat(),
                "vector": [0.0] * EMBEDDING_DIM
            }]
            self.table = self.db.create_table(table_name, initial_data)
            logger.info("Created new conversations table")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Ollama
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        try:
            response = self.ollama_client.embeddings(
                model=EMBEDDING_MODEL,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * EMBEDDING_DIM
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str
    ) -> str:
        """
        Add message to conversation memory
        
        Args:
            conversation_id: Conversation session ID
            role: Message role (user/assistant)
            content: Message content
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())[:8]
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Create record
        record = {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "vector": embedding
        }
        
        # Add to table
        self.table.add([record])
        
        logger.debug(f"Added message {message_id} to conversation {conversation_id}")
        return message_id
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar messages in conversation history
        
        Args:
            query: Search query text
            limit: Maximum results to return
            conversation_id: Optional filter by conversation
            
        Returns:
            List of matching messages with scores
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search in LanceDB
        results = self.table.search(query_embedding).limit(limit).to_list()
        
        # Filter by conversation if specified
        if conversation_id:
            results = [r for r in results if r.get("conversation_id") == conversation_id]
        
        # Filter out init record
        results = [r for r in results if r.get("id") != "init"]
        
        # Format results
        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("id"),
                "conversation_id": r.get("conversation_id"),
                "role": r.get("role"),
                "content": r.get("content"),
                "timestamp": r.get("timestamp"),
                "similarity_score": 1 - r.get("_distance", 0)  # Convert distance to similarity
            })
        
        return formatted
    
    def get_conversation_context(
        self, 
        conversation_id: str, 
        around_message_id: Optional[str] = None,
        window: int = 3
    ) -> str:
        """
        Get formatted conversation context
        
        Args:
            conversation_id: Conversation ID
            around_message_id: Center context around this message
            window: Number of messages before/after
            
        Returns:
            Formatted context string
        """
        # Get all messages from conversation
        all_results = self.table.search([0.0] * EMBEDDING_DIM).limit(1000).to_list()
        
        # Filter by conversation
        messages = [
            r for r in all_results 
            if r.get("conversation_id") == conversation_id and r.get("id") != "init"
        ]
        
        # Sort by timestamp
        messages.sort(key=lambda x: x.get("timestamp", ""))
        
        if not messages:
            return "No conversation history found."
        
        # Format context
        context_parts = []
        for msg in messages[-window*2:]:  # Last N messages
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            context_parts.append(f"{role}: {content}")
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        all_results = self.table.search([0.0] * EMBEDDING_DIM).limit(10000).to_list()
        
        # Filter out init
        messages = [r for r in all_results if r.get("id") != "init"]
        
        # Count by conversation
        conversations = {}
        for msg in messages:
            conv_id = msg.get("conversation_id", "unknown")
            conversations[conv_id] = conversations.get(conv_id, 0) + 1
        
        return {
            "total_messages": len(messages),
            "total_conversations": len(conversations),
            "messages_per_conversation": conversations
        }
