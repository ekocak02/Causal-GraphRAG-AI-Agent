from typing import List, Dict, Any, Optional
import logging

from memory.conversation_store import ConversationStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatMemory:
    """
    High-level chat memory interface
    
    Features:
    - Conversation management
    - Title generation via orchestrator
    - Message history with context
    """
    
    def __init__(self, store_path: str = "conversations"):
        """
        Initialize chat memory
        
        Args:
            store_path: Path to conversation storage
        """
        self.store = ConversationStore(store_path)
        self._orchestrator = None
    
    @property
    def orchestrator(self):
        """Lazy load orchestrator to avoid circular imports"""
        if self._orchestrator is None:
            from agents.orchestrator import OrchestratorAgent
            self._orchestrator = OrchestratorAgent()
        return self._orchestrator
    
    def new_conversation(self, first_message: Optional[str] = None) -> str:
        """
        Start a new conversation
        
        Args:
            first_message: Optional first user message for title generation
            
        Returns:
            Conversation ID
        """
        title = None
        if first_message:
            title = self.generate_title([{"role": "user", "content": first_message}])
        
        conversation_id = self.store.create_conversation(title)
        logger.info(f"Created new conversation: {conversation_id}")
        
        return conversation_id
    
    def add_user_message(self, conversation_id: str, content: str) -> None:
        """
        Add user message to conversation
        
        Args:
            conversation_id: Conversation ID
            content: Message content
        """
        self.store.save_message(conversation_id, "user", content)
        
        # Auto-generate title if this is the first message
        messages = self.store.get_messages(conversation_id)
        if len(messages) == 1:
            title = self.generate_title(messages)
            self.store.update_title(conversation_id, title)
    
    def add_assistant_message(
        self, 
        conversation_id: str, 
        content: str,
        agent_responses: Optional[List[Dict]] = None,
        visualizations: Optional[List[str]] = None
    ) -> None:
        """
        Add assistant message to conversation
        
        Args:
            conversation_id: Conversation ID
            content: Message content
            agent_responses: Optional agent response data for debugging
            visualizations: Optional visualization file paths
        """
        self.store.save_message(
            conversation_id, 
            "assistant", 
            content,
            agent_responses=agent_responses,
            visualizations=visualizations
        )
    
    def get_history(
        self, 
        conversation_id: str, 
        last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            conversation_id: Conversation ID
            last_n: Optional limit to last N messages
            
        Returns:
            List of messages
        """
        messages = self.store.get_messages(conversation_id)
        
        if last_n is not None:
            messages = messages[-last_n:]
        
        return messages
    
    def get_context_for_llm(
        self, 
        conversation_id: str, 
        max_messages: int = 10
    ) -> str:
        """
        Get formatted conversation context for LLM prompt
        
        Args:
            conversation_id: Conversation ID
            max_messages: Maximum messages to include
            
        Returns:
            Formatted context string
        """
        messages = self.get_history(conversation_id, last_n=max_messages)
        
        context_parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n\n".join(context_parts)
    
    def generate_title(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate conversation title using orchestrator
        
        Args:
            messages: List of messages (at least first user message)
            
        Returns:
            Generated title
        """
        if not messages:
            return "New Conversation"
        
        # Get first few messages for context
        context = []
        for msg in messages[:3]:
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            context.append(f"{msg.get('role', 'user')}: {content}")
        
        context_str = "\n".join(context)
        
        prompt = f"""Based on the following conversation start, generate a short, descriptive title (max 6 words).

CONVERSATION:
{context_str}

OUTPUT as JSON:
{{"title": "your title here"}}

RULES:
- Title should be concise and descriptive
- Focus on the main topic or question
- No quotes or special characters
- Maximum 6 words
"""
        
        try:
            response = self.orchestrator.call_llm(prompt, temperature=0.3, format="json")
            return response.get("title", "New Conversation")[:50]
        except Exception as e:
            logger.warning(f"Failed to generate title: {e}")
            if messages:
                first_content = messages[0].get("content", "")[:50]
                return first_content + "..." if len(first_content) == 50 else first_content
            return "New Conversation"
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversations
        
        Returns:
            List of conversation metadata
        """
        return self.store.list_conversations()
    
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation
        
        Args:
            conversation_id: Conversation ID
        """
        self.store.delete_conversation(conversation_id)
        logger.info(f"Deleted conversation: {conversation_id}")
    
    def summarize_conversation(self, conversation_id: str) -> str:
        """
        Generate a summary of the conversation
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Summary string
        """
        messages = self.get_history(conversation_id)
        
        if not messages:
            return "Empty conversation"
        
        # Format messages for summary
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:500]
            formatted.append(f"{role}: {content}")
        
        conversation_text = "\n\n".join(formatted)
        
        prompt = f"""Summarize the following conversation in 2-3 sentences. Focus on the main topics discussed and any conclusions reached.

CONVERSATION:
{conversation_text}

OUTPUT as JSON:
{{"summary": "your summary here"}}
"""
        
        try:
            response = self.orchestrator.call_llm(prompt, temperature=0.3, format="json")
            return response.get("summary", "Unable to generate summary")
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return f"Conversation with {len(messages)} messages"
