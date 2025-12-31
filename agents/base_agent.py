from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import json
import ollama

from agents.models.schemas import AgentTask, AgentResponse
from agents.config import OLLAMA_HOST, OLLAMA_TIMEOUT, MAX_AGENT_ITERATIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialist agents
    
    Each agent must implement:
    - agent_type: Agent identifier
    - system_prompt: System prompt for LLM
    - available_tools: List of tools this agent can use
    - _process_task: Core task processing logic
    """
    
    def __init__(self, model_name: str):
        """
        Initialize agent with Ollama model
        
        Args:
            model_name: Ollama model to use (e.g., 'qwen2.5:7b')
        """
        self.model_name = model_name
        self.ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.iteration_count = 0
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Agent type identifier"""
        pass
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for LLM"""
        pass
    
    @property
    @abstractmethod
    def available_tools(self) -> List[Any]:
        """List of available tool instances"""
        pass
    
    @abstractmethod
    def _process_task(self, instruction: str, context: Optional[Dict[str, Any]]) -> Any:
        """
        Core task processing logic
        
        Args:
            instruction: Task instruction from orchestrator
            context: Additional context
            
        Returns:
            Task result (format depends on agent)
        """
        pass
    
    def execute_task(self, task: AgentTask) -> AgentResponse:
        """
        Execute assigned task with error handling
        
        Args:
            task: AgentTask object
            
        Returns:
            AgentResponse with result or error
        """
        self.iteration_count = 0
        
        try:
            self.logger.info(f"[{self.agent_type.upper()}] Executing task: {task.task_id}")
            self.logger.debug(f"Instruction: {task.instruction}")
            
            result = self._process_task(task.instruction, task.context)
            
            return AgentResponse(
                task_id=task.task_id,
                agent_type=self.agent_type,
                success=True,
                result=result,
                error=None,
                tools_used=[tool.name for tool in self.available_tools],
                iterations=self.iteration_count
            )
            
        except Exception as e:
            error_msg = f"Task failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return AgentResponse(
                task_id=task.task_id,
                agent_type=self.agent_type,
                success=False,
                result=None,
                error=error_msg,
                tools_used=[],
                iterations=self.iteration_count
            )
    
    def call_llm(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Call Ollama LLM with structured output
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0-1.0)
            format: Output format ("json" or "text")
            
        Returns:
            Parsed response (dict if json, str if text)
        """
        self.iteration_count += 1
        
        if self.iteration_count > MAX_AGENT_ITERATIONS:
            raise RuntimeError(f"Max iterations ({MAX_AGENT_ITERATIONS}) exceeded")
        
        try:
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": 2000  
                },
                format=format if format == "json" else None
            )
            
            content = response['message']['content']
            
            if format == "json":
                # Parse JSON response
                return json.loads(content)
            else:
                return {"text": content}
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            raise ValueError(f"LLM returned invalid JSON: {content}")
        
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Any]:
        """
        Get tool instance by name
        
        Args:
            tool_name: Tool name (e.g., "semantic_search")
            
        Returns:
            Tool instance or None if not found
        """
        for tool in self.available_tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def format_tool_descriptions(self) -> str:
        """
        Format tool descriptions for LLM prompt
        
        Returns:
            Markdown-formatted tool descriptions
        """
        descriptions = []
        for tool in self.available_tools:
            descriptions.append(f"### {tool.name}")
            descriptions.append(tool.description)
            descriptions.append(f"**Input Schema:** {json.dumps(tool.input_schema.schema(), indent=2)}")
            descriptions.append("")
        
        return "\n".join(descriptions)