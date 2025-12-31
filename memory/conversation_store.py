import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


class ConversationStore:
    """
    Simple JSON-based storage for conversations
    
    Structure:
    conversations/
        {id}/
            metadata.json  - {id, title, created_at, updated_at, message_count}
            messages.json  - [{role, content, timestamp, agent_responses}]
    """
    
    def __init__(self, base_path: str = "conversations"):
        """
        Initialize conversation store
        
        Args:
            base_path: Base directory for storing conversations
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation
        
        Args:
            title: Optional title (auto-generated if not provided)
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())[:8]
        conv_path = self.base_path / conversation_id
        conv_path.mkdir(parents=True, exist_ok=True)
        
        now = datetime.now().isoformat()
        
        metadata = {
            "id": conversation_id,
            "title": title or f"Conversation {conversation_id}",
            "created_at": now,
            "updated_at": now,
            "message_count": 0
        }
        
        self._save_json(conv_path / "metadata.json", metadata)
        self._save_json(conv_path / "messages.json", [])
        
        return conversation_id
    
    def save_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        agent_responses: Optional[List[Dict]] = None,
        visualizations: Optional[List[str]] = None
    ) -> None:
        """
        Save a message to conversation
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant/system)
            content: Message content
            agent_responses: Optional agent response data
            visualizations: Optional visualization paths
        """
        conv_path = self.base_path / conversation_id
        
        if not conv_path.exists():
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Load existing messages
        messages = self._load_json(conv_path / "messages.json")
        
        # Add new message
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "agent_responses": agent_responses,
            "visualizations": visualizations
        }
        messages.append(message)
        
        # Save messages
        self._save_json(conv_path / "messages.json", messages)
        
        # Update metadata
        metadata = self._load_json(conv_path / "metadata.json")
        metadata["updated_at"] = datetime.now().isoformat()
        metadata["message_count"] = len(messages)
        self._save_json(conv_path / "metadata.json", metadata)
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages from conversation
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of messages
        """
        conv_path = self.base_path / conversation_id
        
        if not conv_path.exists():
            raise ValueError(f"Conversation {conversation_id} not found")
        
        return self._load_json(conv_path / "messages.json")
    
    def get_metadata(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation metadata
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Metadata dict
        """
        conv_path = self.base_path / conversation_id
        
        if not conv_path.exists():
            raise ValueError(f"Conversation {conversation_id} not found")
        
        return self._load_json(conv_path / "metadata.json")
    
    def update_title(self, conversation_id: str, title: str) -> None:
        """
        Update conversation title
        
        Args:
            conversation_id: Conversation ID
            title: New title
        """
        conv_path = self.base_path / conversation_id
        
        if not conv_path.exists():
            raise ValueError(f"Conversation {conversation_id} not found")
        
        metadata = self._load_json(conv_path / "metadata.json")
        metadata["title"] = title
        metadata["updated_at"] = datetime.now().isoformat()
        self._save_json(conv_path / "metadata.json", metadata)
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversations
        
        Returns:
            List of conversation metadata sorted by updated_at (newest first)
        """
        conversations = []
        
        for conv_dir in self.base_path.iterdir():
            if conv_dir.is_dir():
                metadata_path = conv_dir / "metadata.json"
                if metadata_path.exists():
                    conversations.append(self._load_json(metadata_path))
        
        # Sort by updated_at descending
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation
        
        Args:
            conversation_id: Conversation ID
        """
        conv_path = self.base_path / conversation_id
        
        if conv_path.exists():
            import shutil
            shutil.rmtree(conv_path)
    
    def _save_json(self, path: Path, data: Any) -> None:
        """Save data to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_json(self, path: Path) -> Any:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
