"""
Memory Tools: Conversation History Search
"""

from typing import List, Dict, Any

from agents.tools.base_tool import BaseTool
from memory.conversation_rag import ConversationRAG


class ConversationSearchInput:
    """Input schema for conversation search"""
    query: str
    limit: int = 5


class ConversationSearchTool(BaseTool):
    """
    Search past conversations for relevant context
    """
    
    def __init__(self):
        super().__init__()
        self._rag = None  # Lazy initialization
    
    @property
    def rag(self) -> ConversationRAG:
        """Lazy load RAG to avoid circular imports"""
        if self._rag is None:
            self._rag = ConversationRAG()
        return self._rag
    
    @property
    def name(self) -> str:
        return "search_conversation_history"
    
    @property
    def description(self) -> str:
        return """Search past conversation history for relevant context.
        Use this when user asks about previous discussions, past queries, or says things like:
        - "What did we discuss about X?"
        - "In our earlier conversation..."
        - "Based on what we talked about..."
        - "Remind me about..."
        
        Returns the most relevant past messages based on semantic similarity."""
    
    @property
    def input_schema(self) -> type:
        return ConversationSearchInput
    
    def _execute(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history
        
        Args:
            query: Search query text
            limit: Maximum results
            
        Returns:
            List of relevant past messages
        """
        results = self.rag.search(query, limit=limit)
        
        if not results:
            return {
                "found": False,
                "message": "No relevant past conversations found.",
                "results": []
            }
        
        # Format for LLM consumption
        formatted_results = []
        for r in results:
            formatted_results.append({
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "relevance": f"{r['similarity_score']:.2f}"
            })
        
        return {
            "found": True,
            "message": f"Found {len(results)} relevant past messages.",
            "results": formatted_results
        }


# Singleton instance for shared usage
_conversation_rag_instance = None

def get_conversation_rag() -> ConversationRAG:
    """Get shared ConversationRAG instance"""
    global _conversation_rag_instance
    if _conversation_rag_instance is None:
        _conversation_rag_instance = ConversationRAG()
    return _conversation_rag_instance
