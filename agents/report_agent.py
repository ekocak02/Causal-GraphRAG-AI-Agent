from typing import List, Dict, Any, Optional

from agents.base_agent import BaseAgent
from agents.models.schemas import AgentResponse, FinalReport
from agents.prompts.system_prompts import REPORT_AGENT_SYSTEM_PROMPT
from agents.config import REPORT_MODEL


class ReportAgent(BaseAgent):
    """
    Final agent that synthesizes all specialist findings
    
    Capabilities:
    - Aggregate results from multiple agents
    - Create structured, user-friendly reports
    - Assess overall confidence
    - Generate recommendations
    """
    
    def __init__(self):
        super().__init__(model_name=REPORT_MODEL)
    
    @property
    def agent_type(self) -> str:
        return "report"
    
    @property
    def system_prompt(self) -> str:
        return REPORT_AGENT_SYSTEM_PROMPT
    
    @property
    def available_tools(self) -> List[Any]:
        return []
    
    def create_report(
        self,
        user_query: str,
        agent_responses: List[AgentResponse],
        report_instructions: str
    ) -> FinalReport:
        """
        Create final report from agent responses
        
        Args:
            user_query: Original user query
            agent_responses: List of responses from specialist agents
            report_instructions: Instructions from Orchestrator
            
        Returns:
            FinalReport with structured findings
        """
        self.logger.info(f"Creating report for query: {user_query}")
        
        # Format agent responses for LLM
        formatted_responses = self._format_agent_responses(agent_responses)
        
        # Prepare prompt
        prompt = f"""
USER QUERY: {user_query}

REPORT INSTRUCTIONS: {report_instructions}

AGENT FINDINGS:
{formatted_responses}

TASK: Create a comprehensive, user-friendly report that answers the user's query.

OUTPUT as JSON:
{{
    "summary": "2-3 sentence executive summary answering the query",
    "findings": [
        {{
            "category": "Graph Analysis|Causal Analysis|Statistical Analysis|Risk Assessment",
            "key_points": ["point 1", "point 2", ...],
            "supporting_data": {{...}}
        }}
    ],
    "recommendations": ["actionable recommendation 1", "..."],
    "confidence_score": 0.85  // 0.0-1.0
}}

GUIDELINES:
1. Start with a clear, direct answer to the user's query
2. Organize findings by category (matching agent types)
3. Use plain language, explain technical terms
4. Quantify findings with specific numbers
5. Cite sources (e.g., "Tigramite analysis shows...", "XGBoost model predicts...")
6. Acknowledge gaps or uncertainties
7. Provide actionable recommendations if relevant
8. Assess confidence based on:
   - Data quality and completeness
   - Statistical significance
   - Agreement between different analyses
   - Model confidence scores
"""
        
        response = self.call_llm(prompt, temperature=0.3, format="json")
        
        # Extract visualization paths from agent responses
        visualization_paths = []
        for agent_resp in agent_responses:
            if agent_resp.success and isinstance(agent_resp.result, dict):
                # Look for plot_path in result
                result = agent_resp.result
                if "result" in result and isinstance(result["result"], dict):
                    plot_path = result["result"].get("plot_path")
                    if plot_path:
                        visualization_paths.append(plot_path)
        
        # List agents that contributed
        agents_used = [resp.agent_type for resp in agent_responses if resp.success]
        
        # Create FinalReport
        report = FinalReport(
            user_query=user_query,
            summary=response["summary"],
            findings=response["findings"],
            visualizations=visualization_paths,
            recommendations=response.get("recommendations"),
            confidence_score=response["confidence_score"],
            agents_used=agents_used
        )
        
        self.logger.info(f"Report created: {len(report.findings)} findings, confidence={report.confidence_score:.2f}")
        
        return report
    
    def _format_agent_responses(self, responses: List[AgentResponse]) -> str:
        """
        Format agent responses into readable text for LLM
        
        Args:
            responses: List of agent responses
            
        Returns:
            Formatted string
        """
        formatted = []
        
        for resp in responses:
            section = f"\n{'='*60}\n"
            section += f"AGENT: {resp.agent_type.upper()}\n"
            section += f"Status: {'✓ SUCCESS' if resp.success else '✗ FAILED'}\n"
            
            if resp.success:
                section += f"Tools Used: {', '.join(resp.tools_used)}\n"
                section += f"Iterations: {resp.iterations}\n"
                section += "\nRESULT:\n"
                section += self._format_result(resp.result)
            else:
                section += f"Error: {resp.error}\n"
            
            formatted.append(section)
        
        return "\n".join(formatted)
    
    def _format_result(self, result: Any, indent: int = 0) -> str:
        """
        Recursively format result data for readability
        
        Args:
            result: Result data (dict, list, or primitive)
            indent: Indentation level
            
        Returns:
            Formatted string
        """
        prefix = "  " * indent
        
        if isinstance(result, dict):
            lines = []
            for key, value in result.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._format_result(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
            return "\n".join(lines)
        
        elif isinstance(result, list):
            if not result:
                return f"{prefix}[]"
            
            lines = []
            for i, item in enumerate(result[:5]):
                if isinstance(item, dict):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(self._format_result(item, indent + 1))
                else:
                    lines.append(f"{prefix}[{i}]: {item}")
            
            if len(result) > 5:
                lines.append(f"{prefix}... ({len(result) - 5} more items)")
            
            return "\n".join(lines)
        
        else:
            return f"{prefix}{result}"
    
    def _process_task(self, instruction: str, context: Optional[Dict[str, Any]]) -> FinalReport:
        """
        Process report creation task
        
        Args:
            instruction: User query (from context)
            context: Must contain "agent_responses" and "report_instructions"
            
        Returns:
            FinalReport
        """
        if not context:
            raise ValueError("Context must contain agent_responses and report_instructions")
        
        user_query = context.get("user_query", instruction)
        agent_responses = context.get("agent_responses", [])
        report_instructions = context.get("report_instructions", "Create a comprehensive report")
        
        return self.create_report(user_query, agent_responses, report_instructions)