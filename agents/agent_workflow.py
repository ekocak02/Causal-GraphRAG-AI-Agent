import logging
from typing import List
from datetime import datetime

from agents import (
    OrchestratorAgent,
    GraphAgent,
    CausalAgent,
    StatisticalAgent,
    RiskAgent,
    ReportAgent
)
from agents.models.schemas import (
    AgentTask, 
    AgentResponse, 
    FinalReport,
    CommunicationLog,
    AgentWorkflowResult
)
from agents.config import MAX_TOOL_RETRIES, ERROR_FEEDBACK_ENABLED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentWorkflow:
    """
    Main workflow orchestrator for multi-agent system
    
    Manages:
    - Agent initialization
    - Task execution
    - Error handling and retries
    - Final report generation
    """
    
    def __init__(self):
        """Initialize all agents"""
        logger.info("Initializing agent workflow...")
        
        self.orchestrator = OrchestratorAgent()
        self.graph_agent = GraphAgent()
        self.causal_agent = CausalAgent()
        self.statistical_agent = StatisticalAgent()
        self.risk_agent = RiskAgent()
        self.report_agent = ReportAgent()
        
        # Agent registry
        self.agents = {
            "graph": self.graph_agent,
            "causal": self.causal_agent,
            "statistical": self.statistical_agent,
            "risk": self.risk_agent
        }
        
        logger.info(f"Initialized {len(self.agents)} specialist agents")
    
    def execute_query(self, user_query: str) -> AgentWorkflowResult:
        """
        Execute user query through multi-agent system
        
        Args:
            user_query: User's question or request
            
        Returns:
            AgentWorkflowResult with report and communication logs
        """
        # Initialize communication logs
        comm_logs: List[CommunicationLog] = []
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EXECUTING QUERY: {user_query}")
        logger.info(f"{'='*80}\n")
        
        # Log user input
        comm_logs.append(CommunicationLog(
            step_type="user_input",
            content=user_query,
            metadata={"timestamp": datetime.now().isoformat()}
        ))
        
        # Create execution plan
        logger.info("[STEP 1] Creating execution plan:")
        plan = self.orchestrator.create_plan(user_query)
        
        # Log orchestrator plan
        task_list = [
            {"agent": t.agent_type, "instruction": t.instruction, "task_id": t.task_id}
            for t in plan.tasks
        ]
        comm_logs.append(CommunicationLog(
            step_type="orchestrator_plan",
            agent_type="orchestrator",
            content=f"Created plan with {len(plan.tasks)} tasks",
            metadata={
                "tasks": task_list,
                "execution_strategy": plan.execution_strategy,
                "report_instructions": getattr(self.orchestrator, 'report_instructions', '')
            }
        ))
        
        logger.info(f"Plan: {len(plan.tasks)} tasks, strategy={plan.execution_strategy}")
        for i, task in enumerate(plan.tasks, 1):
            logger.info(f"  Task {i}: [{task.agent_type}] {task.instruction[:60]}")
        
        # Execute tasks with logging
        logger.info("\n[STEP 2] Executing tasks:")
        agent_responses = self._execute_tasks_with_logging(plan.tasks, comm_logs)
        
        # Generate final report
        logger.info("\n[STEP 3] Generating final report:")
        
        # Log report agent being called
        comm_logs.append(CommunicationLog(
            step_type="agent_task",
            agent_type="report",
            content="Synthesizing findings from all agents",
            metadata={
                "input_agents": [r.agent_type for r in agent_responses],
                "successful_responses": sum(1 for r in agent_responses if r.success)
            }
        ))
        
        report = self.report_agent.create_report(
            user_query=user_query,
            agent_responses=agent_responses,
            report_instructions=self.orchestrator.report_instructions
        )
        
        # Log final report
        comm_logs.append(CommunicationLog(
            step_type="report",
            agent_type="report",
            content=report.summary,
            metadata={
                "confidence_score": report.confidence_score,
                "findings_count": len(report.findings),
                "visualizations_count": len(report.visualizations),
                "recommendations": report.recommendations
            }
        ))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"QUERY COMPLETED")
        logger.info(f"  Agents Used: {', '.join(report.agents_used)}")
        logger.info(f"  Confidence: {report.confidence_score:.2f}")
        logger.info(f"  Findings: {len(report.findings)}")
        logger.info(f"  Visualizations: {len(report.visualizations)}")
        logger.info(f"{'='*80}\n")
        
        return AgentWorkflowResult(
            report=report,
            communication_logs=comm_logs
        )
    
    def _execute_tasks(self, tasks: List[AgentTask]) -> List[AgentResponse]:
        """
        Execute list of tasks with error handling
        
        Args:
            tasks: List of agent tasks
            
        Returns:
            List of agent responses
        """
        responses = []
        
        for i, task in enumerate(tasks, 1):
            logger.info(f"\nTask {i}/{len(tasks)}: [{task.agent_type.upper()}] {task.task_id}")
            logger.info(f"  Instruction: {task.instruction}")
            
            response = self._execute_task_with_retry(task)
            responses.append(response)
            
            if response.success:
                logger.info(f" Task completed successfully")
            else:
                logger.warning(f" Task failed: {response.error}")
        
        return responses
    
    def _execute_tasks_with_logging(
        self, 
        tasks: List[AgentTask], 
        comm_logs: List[CommunicationLog]
    ) -> List[AgentResponse]:
        """
        Execute list of tasks with communication logging
        
        Args:
            tasks: List of agent tasks
            comm_logs: Communication logs list to append to
            
        Returns:
            List of agent responses
        """
        responses = []
        
        for i, task in enumerate(tasks, 1):
            logger.info(f"\nTask {i}/{len(tasks)}: [{task.agent_type.upper()}] {task.task_id}")
            logger.info(f"  Instruction: {task.instruction}")
            
            # Log task being sent to agent
            comm_logs.append(CommunicationLog(
                step_type="agent_task",
                agent_type=task.agent_type,
                content=task.instruction,
                metadata={
                    "task_id": task.task_id,
                    "priority": task.priority,
                    "context": task.context
                }
            ))
            
            response = self._execute_task_with_retry(task)
            responses.append(response)
            
            if response.success:
                logger.info(f" Task completed successfully")
                # Log successful response
                comm_logs.append(CommunicationLog(
                    step_type="agent_response",
                    agent_type=task.agent_type,
                    content=f"Task completed successfully",
                    metadata={
                        "task_id": task.task_id,
                        "success": True,
                        "tools_used": response.tools_used,
                        "iterations": response.iterations,
                        "result_summary": self._summarize_result(response.result)
                    }
                ))
            else:
                logger.warning(f" Task failed: {response.error}")
                # Log error response
                comm_logs.append(CommunicationLog(
                    step_type="error",
                    agent_type=task.agent_type,
                    content=response.error or "Unknown error",
                    metadata={
                        "task_id": task.task_id,
                        "success": False,
                        "tools_used": response.tools_used,
                        "iterations": response.iterations
                    }
                ))
        
        return responses
    
    def _summarize_result(self, result: any) -> str:
        """Create a brief summary of agent result for logging"""
        if result is None:
            return "No result"
        if isinstance(result, dict):
            keys = list(result.keys())[:5]
            return f"Dict with keys: {keys}"
        if isinstance(result, list):
            return f"List with {len(result)} items"
        if isinstance(result, str):
            return result[:200] + "..." if len(result) > 200 else result
        return str(type(result).__name__)
    
    def _execute_task_with_retry(self, task: AgentTask) -> AgentResponse:
        """
        Execute task with retry logic
        
        Args:
            task: Agent task
            
        Returns:
            AgentResponse
        """
        # Get agent
        agent = self.agents.get(task.agent_type)
        if not agent:
            error_msg = f"Unknown agent type: {task.agent_type}"
            logger.error(error_msg)
            return AgentResponse(
                task_id=task.task_id,
                agent_type=task.agent_type,
                success=False,
                result=None,
                error=error_msg,
                tools_used=[],
                iterations=0
            )
        
        # Execute with retries
        for attempt in range(MAX_TOOL_RETRIES + 1):
            try:
                response = agent.execute_task(task)
                
                if response.success:
                    return response
                
                # Handle failure
                if ERROR_FEEDBACK_ENABLED and attempt < MAX_TOOL_RETRIES:
                    logger.warning(f"  Attempt {attempt + 1} failed, trying corrective action...")
                    
                    # Get corrective task from orchestrator
                    corrective_task = self.orchestrator.handle_agent_failure(task, response.error)
                    
                    if corrective_task:
                        task = corrective_task 
                    else:
                        # Orchestrator says to skip
                        return response
                else:
                    # Max retries reached or feedback disabled
                    return response
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                if attempt == MAX_TOOL_RETRIES:
                    return AgentResponse(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        success=False,
                        result=None,
                        error=error_msg,
                        tools_used=[],
                        iterations=0
                    )
        
        # Should not reach here
        return AgentResponse(
            task_id=task.task_id,
            agent_type=task.agent_type,
            success=False,
            result=None,
            error="Max retries exceeded",
            tools_used=[],
            iterations=0
        )
    
    def format_report_for_display(self, report: FinalReport) -> str:
        """
        Format final report for console/UI display
        
        Args:
            report: FinalReport object
            
        Returns:
            Formatted string
        """
        output = []
        
        output.append("\n" + "-"*30)
        output.append("FINANCIAL MARKET ANALYSIS REPORT")
        output.append("-"*30)
        
        output.append(f"\nQuery: {report.user_query}")
        output.append(f"Confidence: {report.confidence_score:.1%}")
        output.append(f"Agents: {', '.join(report.agents_used)}")
        
        output.append("\n" + "="*30)
        output.append("EXECUTIVE SUMMARY")
        output.append("="*30)
        output.append(report.summary)
        
        output.append("\n" + "-"*30)
        output.append("DETAILED FINDINGS")
        output.append("-"*30)
        
        for i, finding in enumerate(report.findings, 1):
            output.append(f"\n{i}. {finding['category']}")
            for point in finding.get('key_points', []):
                output.append(f"   • {point}")
        
        if report.recommendations:
            output.append("\n" + "-"*30)
            output.append("RECOMMENDATIONS")
            output.append("-"*30)
            for i, rec in enumerate(report.recommendations, 1):
                output.append(f"{i}. {rec}")
        
        if report.visualizations:
            output.append("\n" + "-"*30)
            output.append("VISUALIZATIONS")
            output.append("-"*30)
            for viz in report.visualizations:
                output.append(f"  📊 {viz}")
        
        output.append("\n" + "="*30 + "\n")
        
        return "\n".join(output)


# Convenience function for single query execution
def run_query(user_query: str) -> FinalReport:
    """
    Convenience function to run a single query
    
    Args:
        user_query: User's question
        
    Returns:
        FinalReport
    """
    workflow = AgentWorkflow()
    return workflow.execute_query(user_query)


if __name__ == "__main__":
    # Example usage
    query = input("Enter your query: ")
    
    workflow = AgentWorkflow()
    report = workflow.execute_query(query)
    print(workflow.format_report_for_display(report))