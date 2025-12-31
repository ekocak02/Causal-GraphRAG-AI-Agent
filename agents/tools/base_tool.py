from abc import ABC, abstractmethod
from typing import Any, Dict
import logging
from pydantic import BaseModel

from agents.models.schemas import ToolResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools
    
    Each tool must implement:
    - name: Tool identifier
    - description: What the tool does
    - input_schema: Pydantic model for input validation
    - _execute: Core tool logic
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM"""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Pydantic model for input validation"""
        pass
    
    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """
        Core tool execution logic
        Subclasses must implement this
        """
        pass
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool with error handling and validation
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult: Standardized result wrapper
        """
        try:
            # Validate input
            validated_input = self.input_schema(**kwargs)
            
            self.logger.info(f"Executing {self.name} with params: {validated_input.dict()}")
            
            # Execute core logic
            result = self._execute(**validated_input.dict())
            
            self.logger.info(f"{self.name} completed successfully")
            
            return ToolResult(
                success=True,
                data=result,
                error=None,
                metadata={"tool": self.name}
            )
            
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ToolResult(
                success=False,
                data=None,
                error=error_msg,
                metadata={"tool": self.name}
            )
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get tool metadata for LLM prompts
        
        Returns:
            Dict with name, description, and input schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.schema()
        }