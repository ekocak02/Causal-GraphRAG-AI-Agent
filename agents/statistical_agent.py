from typing import List, Dict, Any, Optional
from datetime import date

from agents.base_agent import BaseAgent
from agents.tools.statistical_tools import (
    StatisticalSummaryTool,
    CorrelationMapTool,
    DataValidationTool
)
from agents.prompts.system_prompts import STATISTICAL_AGENT_SYSTEM_PROMPT
from agents.config import STATISTICAL_MODEL


class StatisticalAgent(BaseAgent):
    """
    Specialist agent for statistical analysis
    
    Capabilities:
    - Descriptive statistics and visualizations
    - Correlation analysis with heatmaps
    - Data validation (Hurst, Kurtosis, Vol Clustering)
    """
    
    def __init__(self):
        super().__init__(model_name=STATISTICAL_MODEL)
        
        self._summary = StatisticalSummaryTool()
        self._correlation = CorrelationMapTool()
        self._validation = DataValidationTool()
    
    @property
    def agent_type(self) -> str:
        return "statistical"
    
    @property
    def system_prompt(self) -> str:
        return STATISTICAL_AGENT_SYSTEM_PROMPT
    
    @property
    def available_tools(self) -> List[Any]:
        return [self._summary, self._correlation, self._validation]
    
    def _process_task(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process statistical analysis task
        
        Args:
            instruction: Analysis instruction
            context: Additional context
            
        Returns:
            Statistical analysis results
        """
        # Prepare prompt
        prompt = f"""
TASK: {instruction}

AVAILABLE TOOLS:
{self.format_tool_descriptions()}

TOOL SELECTION GUIDE:
- get_statistical_summary: For single variable analysis, trends, distributions
- get_corr_map: For multi-variable correlation analysis (REQUIRES "columns" as a JSON array)
- get_data_val: For data quality validation (realism check)

AVAILABLE COLUMNS FOR ANALYSIS:
- Asset Prices: Asset_01_TEC_Close, Asset_02_IND_Close, Asset_03_FIN_Close, Asset_04_ENE_Close, Asset_05_HEA_Close, ..., Asset_10_HEA_Close
- Macro Indicators: GDP_Growth, Interest_Rate, Unemployment, Logistics, Production

INSTRUCTIONS:
1. Choose the appropriate tool
2. For get_corr_map, you MUST provide "columns" as a JSON array with 2 or more column names from the list above
3. Specify parameters (columns, date ranges, plot types)
4. Explain what statistical property to emphasize

OUTPUT as JSON:
{{
    "tool": "get_statistical_summary|get_corr_map|get_data_val",
    "parameters": {{
        // tool-specific parameters
        // Use ISO format for dates: "2024-01-01"
    }},
    "analysis_focus": "what to emphasize in results"
}}

PARAMETER EXAMPLES:
- Statistical Summary: {{"target_column": "Asset_01_TEC_Close", "plot_type": "line", "x_column": "Date"}}
- Correlation (3 assets): {{"columns": ["Asset_01_TEC_Close", "Asset_02_IND_Close", "Asset_03_FIN_Close"], "method": "pearson"}}
- Correlation (macro vars): {{"columns": ["GDP_Growth", "Interest_Rate", "Unemployment"], "method": "pearson"}}
- Correlation (all assets): {{"columns": ["Asset_01_TEC_Close", "Asset_02_IND_Close", "Asset_03_FIN_Close", "Asset_04_ENE_Close", "Asset_05_HEA_Close"], "method": "pearson"}}
- Validation: {{"start_date": "2024-01-01", "end_date": "2024-12-31"}}
"""
        
        # Get tool decision from LLM
        decision = self.call_llm(prompt, temperature=0.1, format="json")
        
        tool_name = decision["tool"]
        parameters = decision["parameters"]
        
        self.logger.info(f"Using tool: {tool_name}")
        self.logger.debug(f"Parameters: {parameters}")
        
        # Convert date strings to date objects
        if "start_date" in parameters and isinstance(parameters["start_date"], str):
            parameters["start_date"] = date.fromisoformat(parameters["start_date"])
        if "end_date" in parameters and isinstance(parameters["end_date"], str):
            parameters["end_date"] = date.fromisoformat(parameters["end_date"])
        if "target_date" in parameters and isinstance(parameters["target_date"], str):
            parameters["target_date"] = date.fromisoformat(parameters["target_date"])
        
        # Execute tool
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        result = tool.execute(**parameters)
        
        if not result.success:
            raise RuntimeError(f"Tool execution failed: {result.error}")
        
        # Generate insights
        insights = self._generate_insights(tool_name, result.data)
        
        return {
            "tool_used": tool_name,
            "parameters": parameters,
            "result": result.data,
            "insights": insights,
            "analysis_focus": decision["analysis_focus"]
        }
    
    def _generate_insights(self, tool_name: str, data: Dict[str, Any]) -> List[str]:
        """
        Generate statistical insights from tool results
        
        Args:
            tool_name: Name of tool used
            data: Tool result data
            
        Returns:
            List of insight strings
        """
        insights = []
        
        if tool_name == "get_statistical_summary":
            # Check for outliers
            mean = data.get("mean", 0)
            std = data.get("std", 1)
            min_val = data.get("min", 0)
            max_val = data.get("max", 0)
            
            if abs(max_val - mean) > 3 * std:
                insights.append(f"Extreme high outlier detected: {max_val:.2f} (>3σ from mean)")
            if abs(mean - min_val) > 3 * std:
                insights.append(f"Extreme low outlier detected: {min_val:.2f} (>3σ from mean)")
            
            # Check for skewness
            skew = data.get("skewness", 0)
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                insights.append(f"Distribution is highly skewed ({direction}, skewness={skew:.2f})")
            
            # Check for fat tails
            kurtosis = data.get("kurtosis", 3)
            if kurtosis > 5:
                insights.append(f"Heavy-tailed distribution (kurtosis={kurtosis:.2f}, financial data characteristic)")
        
        elif tool_name == "get_corr_map":
            # Find strong correlations
            corr_matrix = data.get("correlation_matrix", {})
            strong_pairs = []
            
            for var1, corrs in corr_matrix.items():
                for var2, corr_val in corrs.items():
                    if var1 < var2 and abs(corr_val) > 0.7:
                        strong_pairs.append((var1, var2, corr_val))
            
            if strong_pairs:
                insights.append(f"Found {len(strong_pairs)} strong correlations (|r| > 0.7)")
                # Show top 3
                strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for var1, var2, corr in strong_pairs[:3]:
                    insights.append(f"  - {var1} ↔ {var2}: r={corr:.3f}")
        
        elif tool_name == "get_data_val":
            # Interpret validation results
            hurst = data.get("hurst_exponent", {}).get("value", 0.5)
            kurtosis = data.get("kurtosis", {}).get("value", 3)
            vol_clustering = data.get("volatility_clustering", {}).get("detected", False)
            passed = data.get("validation_passed", False)
            
            if passed:
                insights.append("Data passes all validation tests (realistic financial simulation)")
            else:
                insights.append("Data fails some validation tests")
            
            insights.append(f"Hurst Exponent: {hurst:.3f} - {data['hurst_exponent']['interpretation']}")
            insights.append(f"Kurtosis: {kurtosis:.2f} - {data['kurtosis']['interpretation']}")
            
            if vol_clustering:
                insights.append("Volatility clustering detected (GARCH effect present)")
            else:
                insights.append("Weak volatility clustering")
        
        return insights