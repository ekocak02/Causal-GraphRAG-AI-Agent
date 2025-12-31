from typing import List, Dict, Any, Optional
import uuid

from agents.base_agent import BaseAgent
from agents.models.schemas import AgentTask, OrchestratorPlan
from agents.prompts.system_prompts import ORCHESTRATOR_SYSTEM_PROMPT
from agents.config import ORCHESTRATOR_MODEL


class OrchestratorAgent(BaseAgent):
    """
    Main coordinator agent that:
    1. Parses user queries
    2. Creates execution plan
    3. Assigns tasks to specialist agents
    4. Handles failures and provides feedback
    5. Instructs Report Agent on final synthesis
    """
    
    def __init__(self):
        super().__init__(model_name=ORCHESTRATOR_MODEL)
        
        # Dangerous patterns to block at input level
        self._blocked_patterns = [
            'DELETE', 'DROP', 'REMOVE', 'TRUNCATE', 'DETACH DELETE',
            'CREATE CONSTRAINT', 'DROP CONSTRAINT', 'CREATE INDEX', 'DROP INDEX'
        ]
        
        # Memory search tool for conversation history
        self._memory_tool = None
    
    @property
    def memory_tool(self):
        """Lazy load memory tool to avoid circular imports"""
        if self._memory_tool is None:
            from agents.tools.memory_tools import ConversationSearchTool
            self._memory_tool = ConversationSearchTool()
        return self._memory_tool
    
    def validate_user_query(self, query: str) -> tuple[bool, str]:
        """
        Validate user query for dangerous operations before processing
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        query_upper = query.upper()
        
        for pattern in self._blocked_patterns:
            if pattern in query_upper:
                return False, f"Query contains blocked operation: {pattern}. Database modifications are not allowed."
        
        return True, ""
    
    def needs_conversation_context(self, query: str) -> bool:
        """Check if query requires past conversation context"""
        context_keywords = [
            'we discussed', 'we talked', 'earlier', 'previous', 'before',
            'last time', 'remind me', 'you said', 'you mentioned',
            'our conversation', 'based on what', 'as you said'
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in context_keywords)
    
    def get_conversation_context(self, query: str) -> str:
        """Get relevant conversation history for query"""
        try:
            result = self.memory_tool._execute(query=query, limit=5)
            if result.get("found") and result.get("results"):
                context_parts = []
                for r in result["results"]:
                    context_parts.append(f"[{r['role'].upper()}]: {r['content']}")
                return "RELEVANT PAST CONVERSATION:\n" + "\n".join(context_parts)
        except Exception as e:
            self.logger.warning(f"Failed to get conversation context: {e}")
        return ""
    
    @property
    def agent_type(self) -> str:
        return "orchestrator"
    
    @property
    def system_prompt(self) -> str:
        return ORCHESTRATOR_SYSTEM_PROMPT
    
    @property
    def available_tools(self) -> List[Any]:
        return [self.memory_tool]
    
    def create_plan(self, user_query: str) -> OrchestratorPlan:
        """
        Create execution plan from user query
        
        Args:
            user_query: User's question/request
            
        Returns:
            OrchestratorPlan with tasks and strategy
        """
        # Validate input for dangerous operations
        is_safe, error_msg = self.validate_user_query(user_query)
        if not is_safe:
            self.logger.warning(f"Blocked dangerous query: {error_msg}")
            # Return empty plan with error
            error_task = AgentTask(
                task_id="blocked",
                agent_type="report",
                instruction=f"The user's query was blocked for security reasons: {error_msg}. Please inform the user that database modification operations are not permitted.",
                priority=1
            )
            return OrchestratorPlan(tasks=[error_task], execution_strategy="sequential")
        
        self.logger.info(f"Creating plan for query: {user_query}")
        
        # Check if user needs past conversation context
        conversation_context = ""
        if self.needs_conversation_context(user_query):
            self.logger.info("Query requires conversation context, searching history...")
            conversation_context = self.get_conversation_context(user_query)
        
        # Prepare prompt
        prompt = f"""
USER QUERY: {user_query}
{f'''
CONVERSATION HISTORY (from past discussions):
{conversation_context}
''' if conversation_context else ''}

TASK: Create an execution plan to answer this query.

ANALYZE:
1. What type of information is needed? (graph data, causal relationships, statistics, risk assessment)
2. Which agents should be involved?
3. What's the logical order of operations?
4. How should the Report Agent structure the final answer?

AVAILABLE AGENTS:
- graph: Neo4j queries (events, relationships)
- causal: Tigramite and EconML analysis
- statistical: Descriptive statistics, correlations, data validation
- risk: Crisis probability and volatility predictions

OUTPUT your plan as JSON following this exact structure:
{{
    "tasks": [
        {{
            "task_id": "unique_id",
            "agent_type": "graph|causal|statistical|risk",
            "instruction": "Clear, specific instruction for the agent",
            "priority": 1
        }}
    ],
    "execution_strategy": "sequential",
    "report_instructions": "How to structure the final report"
}}

GUIDELINES:
- Be specific in instructions (include date ranges, variables, thresholds)
- Use sequential strategy (tasks depend on each other)
- Always provide report_instructions
- Keep task count reasonable (1-5 tasks)
"""

        response = self.call_llm(prompt, temperature=0.2, format="json")
        
        # Parse into OrchestratorPlan
        tasks = []
        for task_data in response.get("tasks", []):
            if "task_id" not in task_data:
                task_data["task_id"] = str(uuid.uuid4())[:8]
            
            tasks.append(AgentTask(**task_data))
        
        plan = OrchestratorPlan(
            tasks=tasks,
            execution_strategy=response.get("execution_strategy", "sequential")
        )
        
        # Store report instructions for later
        self.report_instructions = response.get("report_instructions", "")
        
        self.logger.info(f"Plan created: {len(tasks)} tasks, strategy={plan.execution_strategy}")
        
        return plan
    
    def handle_agent_failure(
        self, 
        failed_task: AgentTask,
        error_message: str
    ) -> Optional[AgentTask]:
        """
        Handle agent failure and create corrective task
        
        Args:
            failed_task: The task that failed
            error_message: Error message from agent
            
        Returns:
            New corrective task or None if unrecoverable
        """
        self.logger.warning(f"Task {failed_task.task_id} failed: {error_message}")
        
        # Prepare feedback prompt
        prompt = f"""
FAILED TASK:
Agent Type: {failed_task.agent_type}
Instruction: {failed_task.instruction}
Error: {error_message}

TASK: Analyze the failure and create a corrective action.

OPTIONS:
1. Simplify the instruction (e.g., reduce date range, fewer columns)
2. Try alternative approach (e.g., different tool, different query)
3. Skip this task if not critical

OUTPUT as JSON:
{{
    "action": "retry|skip",
    "new_instruction": "modified instruction if retry",
    "reasoning": "brief explanation"
}}
"""
        
        response = self.call_llm(prompt, temperature=0.1, format="json")
        
        action = response.get("action", "skip")
        
        if action == "retry":
            new_task = AgentTask(
                task_id=f"{failed_task.task_id}_retry",
                agent_type=failed_task.agent_type,
                instruction=response["new_instruction"],
                context=failed_task.context,
                priority=failed_task.priority
            )
            self.logger.info(f"Created corrective task: {new_task.task_id}")
            return new_task
        else:
            self.logger.info(f"Skipping failed task: {response.get('reasoning', 'Not critical')}")
            return None
    
    def _process_task(self, instruction: str, context: Optional[Dict[str, Any]]) -> OrchestratorPlan:
        """
        Process task (create plan from user query)
        
        Args:
            instruction: User query
            context: Additional context
            
        Returns:
            OrchestratorPlan
        """
        return self.create_plan(instruction)