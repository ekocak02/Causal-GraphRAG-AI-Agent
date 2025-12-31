from typing import List, Dict, Any, Optional

from agents.base_agent import BaseAgent
from agents.tools.causal_tools import TigramiteTool, EconMLTool, summarize_causal_relationships
from agents.prompts.system_prompts import CAUSAL_AGENT_SYSTEM_PROMPT
from agents.config import CAUSAL_MODEL


class CausalAgent(BaseAgent):
    """
    Specialist agent for causal analysis
    
    Capabilities:
    - Fetch Tigramite causal edges (PCMCI temporal correlations)
    - Fetch EconML inference results (causal treatment effects)
    - Interpret causal relationships
    """
    
    def __init__(self):
        super().__init__(model_name=CAUSAL_MODEL)
        
        self._tigramite = TigramiteTool()
        self._econml = EconMLTool()
    
    @property
    def agent_type(self) -> str:
        return "causal"
    
    @property
    def system_prompt(self) -> str:
        return CAUSAL_AGENT_SYSTEM_PROMPT
    
    @property
    def available_tools(self) -> List[Any]:
        return [self._tigramite, self._econml]
    
    def _process_task(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process causal analysis task
        
        Args:
            instruction: Analysis instruction
            context: Additional context
            
        Returns:
            Causal analysis results
        """
        # Prepare prompt
        prompt = f"""
TASK: {instruction}

AVAILABLE TOOLS:
{self.format_tool_descriptions()}

DATA STRUCTURE REMINDER:
- Tigramite: Shows temporal CORRELATIONS (source→target with lag)
  - strength: -1 to 1 (correlation coefficient)
  - p_value: < 0.05 is statistically significant
  
- EconML: Shows CAUSAL EFFECTS (treatment→outcome)
  - strength: Average Treatment Effect (ATE)
  - confounders: Variables controlled in the analysis
  - policy_tree: Heterogeneous effects by context

INSTRUCTIONS:
1. Decide which tool(s) to use
2. Provide filter parameters (source, target, min_strength, limit)
3. Explain what relationships you're looking for

OUTPUT as JSON:
{{
    "primary_tool": "get_tigramite|get_econml",
    "primary_parameters": {{
        "source": "optional filter",
        "target": "optional filter",
        "min_strength": 0.1,
        "limit": 10
    }},
    "use_secondary": true,  // whether to also use the other tool
    "secondary_parameters": {{}},  // if use_secondary=true
    "analysis_focus": "what to emphasize in interpretation"
}}
"""
        
        # Get tool decision from LLM
        decision = self.call_llm(prompt, temperature=0.1, format="json")
        
        results = {}
        
        # Execute primary tool
        primary_tool_name = decision["primary_tool"]
        primary_tool = self.get_tool_by_name(primary_tool_name)
        
        if not primary_tool:
            raise ValueError(f"Tool '{primary_tool_name}' not found")
        
        primary_result = primary_tool.execute(**decision["primary_parameters"])
        
        if not primary_result.success:
            raise RuntimeError(f"Primary tool failed: {primary_result.error}")
        
        results["primary"] = {
            "tool": primary_tool_name,
            "data": primary_result.data
        }
        
        # Execute secondary tool if requested
        if decision.get("use_secondary", False):
            secondary_tool_name = "get_econml" if primary_tool_name == "get_tigramite" else "get_tigramite"
            secondary_tool = self.get_tool_by_name(secondary_tool_name)
            
            secondary_result = secondary_tool.execute(**decision.get("secondary_parameters", {}))
            
            if secondary_result.success:
                results["secondary"] = {
                    "tool": secondary_tool_name,
                    "data": secondary_result.data
                }
        
        # Create human-readable summary
        tigramite_edges = results["primary"]["data"] if primary_tool_name == "get_tigramite" else \
                         results.get("secondary", {}).get("data", [])
        econml_inferences = results["primary"]["data"] if primary_tool_name == "get_econml" else \
                           results.get("secondary", {}).get("data", [])
        
        summary = summarize_causal_relationships(
            tigramite_edges or [],
            econml_inferences or []
        )
        
        return {
            "results": results,
            "summary": summary,
            "analysis_focus": decision["analysis_focus"],
            "interpretation_hints": self._generate_interpretation_hints(results)
        }
    
    def _generate_interpretation_hints(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate interpretation hints based on results
        
        Args:
            results: Query results
            
        Returns:
            List of interpretation hints
        """
        hints = []
        
        # Check for strong Tigramite correlations
        if "primary" in results and results["primary"]["tool"] == "get_tigramite":
            edges = results["primary"]["data"]
            strong_edges = [e for e in edges if abs(e.strength) > 0.3]
            
            if strong_edges:
                hints.append(f"Found {len(strong_edges)} strong temporal correlations (|strength| > 0.3)")
            
            lagged_edges = [e for e in edges if e.lag > 0]
            if lagged_edges:
                hints.append(f"{len(lagged_edges)} relationships show temporal lag (predictive power)")
        
        # Check for significant EconML effects
        if "primary" in results and results["primary"]["tool"] == "get_econml":
            inferences = results["primary"]["data"]
            strong_effects = [inf for inf in inferences if abs(inf.strength) > 0.01]
            
            if strong_effects:
                hints.append(f"Found {len(strong_effects)} causal effects with ATE > 0.01")
            
            # Check for heterogeneity in policy trees
            complex_trees = [inf for inf in inferences if len(inf.policy_tree) > 100]
            if complex_trees:
                hints.append(f"{len(complex_trees)} relationships show heterogeneous effects (context-dependent)")
        
        return hints