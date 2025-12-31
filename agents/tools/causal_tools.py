import json
from typing import List, Optional

from agents.tools.base_tool import BaseTool
from agents.models.schemas import (
    TigramiteFilterInput,
    EconMLFilterInput,
    TigramiteEdge,
    EconMLInference
)
from agents.config import TIGRAMITE_EDGES_PATH, ECONML_SUMMARY_PATH


class TigramiteTool(BaseTool):
    """
    Fetch and filter Tigramite causal discovery results
    """
    
    def __init__(self):
        super().__init__()
        # Load Tigramite data once at initialization
        with open(TIGRAMITE_EDGES_PATH, 'r') as f:
            self.causal_edges = json.load(f)
    
    @property
    def name(self) -> str:
        return "get_tigramite"
    
    @property
    def description(self) -> str:
        return """Fetch causal edges discovered by Tigramite (PCMCI algorithm).
        Returns source→target relationships with temporal lag, correlation strength, and p-value.
        Use this to understand temporal causal patterns (e.g., 'What causes volatility spikes?')."""
    
    @property
    def input_schema(self) -> type[TigramiteFilterInput]:
        return TigramiteFilterInput
    
    def _execute(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        min_strength: Optional[float] = None,
        max_lag: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[TigramiteEdge]:
        """
        Filter and return Tigramite causal edges
        
        Args:
            source: Filter by source variable (e.g., 'Shock_Active')
            target: Filter by target variable (e.g., 'Vol_Mult')
            min_strength: Minimum absolute strength (correlation)
            max_lag: Maximum temporal lag
            limit: Maximum number of results
            
        Returns:
            List of TigramiteEdge objects
        """
        filtered_edges = []
        
        for edge in self.causal_edges:
            # Apply filters
            if source and edge['source'] != source:
                continue
            if target and edge['target'] != target:
                continue
            if min_strength and abs(edge['strength']) < min_strength:
                continue
            if max_lag is not None and edge['lag'] > max_lag:
                continue
            
            # (p < 0.05)
            if edge['p_value'] < 0.05:
                filtered_edges.append(
                    TigramiteEdge(**edge)
                )
        
        # Sort by absolute strength (strongest first)
        filtered_edges.sort(key=lambda x: abs(x.strength), reverse=True)
        
        # Apply limit
        if limit:
            filtered_edges = filtered_edges[:limit]
        
        return filtered_edges


class EconMLTool(BaseTool):
    """
    Fetch and filter EconML causal inference results
    """
    
    def __init__(self):
        super().__init__()
        # Load EconML data once at initialization
        with open(ECONML_SUMMARY_PATH, 'r') as f:
            self.inference_results = json.load(f)
    
    @property
    def name(self) -> str:
        return "get_econml"
    
    @property
    def description(self) -> str:
        return """Fetch causal inference results from EconML (CausalForestDML).
        Returns treatment effects (ATE), confounders, and policy trees showing heterogeneous effects.
        Use this for causal impact estimation (e.g., 'What's the causal effect of shocks on volatility?')."""
    
    @property
    def input_schema(self) -> type[EconMLFilterInput]:
        return EconMLFilterInput
    
    def _execute(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        min_strength: Optional[float] = None,
        max_lag: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[EconMLInference]:
        """
        Filter and return EconML inference results
        
        Args:
            source: Filter by treatment variable
            target: Filter by outcome variable
            min_strength: Minimum absolute treatment effect (ATE)
            max_lag: Maximum temporal lag
            limit: Maximum number of results
            
        Returns:
            List of EconMLInference objects
        """
        filtered_results = []
        
        for result in self.inference_results:
            # Apply filters
            if source and result['source'] != source:
                continue
            if target and result['target'] != target:
                continue
            if min_strength and abs(result['strength']) < min_strength:
                continue
            if max_lag is not None and result['lag'] > max_lag:
                continue
            
            filtered_results.append(
                EconMLInference(**result)
            )
        
        # Sort by absolute strength (strongest effects first)
        filtered_results.sort(key=lambda x: abs(x.strength), reverse=True)
        
        # Apply limit
        if limit:
            filtered_results = filtered_results[:limit]
        
        return filtered_results


def summarize_causal_relationships(
    tigramite_edges: List[TigramiteEdge],
    econml_inferences: List[EconMLInference]
) -> str:
    """
    Helper function to create human-readable summary of causal findings
    
    Args:
        tigramite_edges: Tigramite causal edges
        econml_inferences: EconML inference results
        
    Returns:
        Markdown-formatted summary
    """
    summary = "## Causal Analysis Summary\n\n"
    
    if tigramite_edges:
        summary += "### Tigramite (Temporal Correlations)\n"
        for edge in tigramite_edges[:5]:  # Top 5
            lag_str = f"lag {edge.lag} days" if edge.lag > 0 else "same day"
            summary += f"- **{edge.source}** → **{edge.target}** ({lag_str})\n"
            summary += f"  - Strength: {edge.strength:.3f}, p-value: {edge.p_value:.4f}\n"
        summary += "\n"
    
    if econml_inferences:
        summary += "### EconML (Causal Effects)\n"
        for inf in econml_inferences[:5]:  # Top 5
            lag_str = f"lag {inf.lag} days" if inf.lag > 0 else "same day"
            summary += f"- **{inf.source}** → **{inf.target}** ({lag_str})\n"
            summary += f"  - Treatment Effect (ATE): {inf.strength:.4f}\n"
            summary += f"  - Confounders: {', '.join(inf.confounders[:3])}{'...' if len(inf.confounders) > 3 else ''}\n"
        summary += "\n"
    
    return summary