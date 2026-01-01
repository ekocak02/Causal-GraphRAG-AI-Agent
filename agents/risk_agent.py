from typing import List, Dict, Any, Optional
from datetime import date

from agents.base_agent import BaseAgent
from agents.tools.ml_tools import CrisisPredictionTool, VolatilityPredictionTool
from agents.prompts.system_prompts import RISK_AGENT_SYSTEM_PROMPT
from agents.config import RISK_MODEL


class RiskAgent(BaseAgent):
    """
    Specialist agent for risk assessment using ML models
    
    Capabilities:
    - Crisis probability prediction (XGBoost)
    - Volatility forecasting (LSTM)
    """
    
    # Valid date range for simulation data
    SIMULATION_START = date(2024, 1, 1)
    SIMULATION_END = date(2033, 12, 31)
    
    def __init__(self):
        super().__init__(model_name=RISK_MODEL)
        
        self._crisis = CrisisPredictionTool()
        self._volatility = VolatilityPredictionTool()
    
    def validate_date(self, target_date: date) -> tuple[bool, str]:
        """
        Validate that target date is within simulation range
        
        Args:
            target_date: Date to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if target_date < self.SIMULATION_START:
            return False, f"Date {target_date} is before simulation start ({self.SIMULATION_START})"
        if target_date > self.SIMULATION_END:
            return False, f"Date {target_date} is after simulation end ({self.SIMULATION_END}). Cannot predict for dates outside 2024-2033 range."
        return True, ""
    
    @property
    def agent_type(self) -> str:
        return "risk"
    
    @property
    def system_prompt(self) -> str:
        return RISK_AGENT_SYSTEM_PROMPT
    
    @property
    def available_tools(self) -> List[Any]:
        return [self._crisis, self._volatility]
    
    def _process_task(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process risk assessment task
        
        Args:
            instruction: Risk assessment instruction
            context: Additional context
            
        Returns:
            Risk assessment results
        """
        # Prepare prompt
        prompt = f"""
TASK: {instruction}

AVAILABLE TOOLS:
{self.format_tool_descriptions()}

TOOL SELECTION:
- get_crisis: Predict crisis probability (XGBoost classifier)
  Output: probability (0-1), risk_level (Low/Medium/High/Critical)
  
- get_volatility: Predict future volatility (LSTM)
  Output: predicted_volatility (annualized), regime (Low/Normal/High/Extreme)

INSTRUCTIONS:
1. Choose the appropriate tool (or both if needed)
2. Specify target date(s) in ISO format: "2024-06-15"
3. For crisis prediction, specify model_choice: "auto" (recommended), "early", or "late"
3. The time frame is between 2024 and 2034.

OUTPUT as JSON:
{{
    "tool": "get_crisis|get_volatility|both",
    "parameters": {{
        "target_date": "2024-06-15",
        "model_choice": "auto"  // only for crisis
    }},
    "additional_dates": [],  // optional: more dates for trend analysis
    "risk_interpretation_focus": "what to emphasize"
}}
"""
        
        # Get tool decision from LLM
        decision = self.call_llm(prompt, temperature=0.1, format="json")
        
        tool_name = decision["tool"]
        parameters = decision["parameters"]
        additional_dates = decision.get("additional_dates", [])
        
        self.logger.info(f"Using tool: {tool_name}")
        
        # Convert date strings to date objects and validate
        if "target_date" in parameters:
            parameters["target_date"] = date.fromisoformat(parameters["target_date"])
            
            # Validate date is within simulation range
            is_valid, error_msg = self.validate_date(parameters["target_date"])
            if not is_valid:
                self.logger.warning(f"Date validation failed: {error_msg}")
                raise ValueError(error_msg)
        
        # Validate additional dates
        validated_additional_dates = []
        for date_str in additional_dates:
            target_date = date.fromisoformat(date_str)
            is_valid, error_msg = self.validate_date(target_date)
            if is_valid:
                validated_additional_dates.append(date_str)
            else:
                self.logger.warning(f"Skipping invalid date {date_str}: {error_msg}")
        additional_dates = validated_additional_dates
        
        results = {}
        
        # Execute primary tool
        if tool_name in ["get_crisis", "both"]:
            crisis_result = self._crisis.execute(**parameters)
            if not crisis_result.success:
                raise RuntimeError(f"Crisis prediction failed: {crisis_result.error}")
            results["crisis"] = crisis_result.data
            
            # Additional dates if requested
            if additional_dates:
                results["crisis_trend"] = []
                for date_str in additional_dates:
                    target_date = date.fromisoformat(date_str)
                    res = self._crisis.execute(target_date=target_date, model_choice=parameters.get("model_choice", "auto"))
                    if res.success:
                        results["crisis_trend"].append(res.data)
        
        if tool_name in ["get_volatility", "both"]:
            vol_result = self._volatility.execute(**parameters)
            if not vol_result.success:
                raise RuntimeError(f"Volatility prediction failed: {vol_result.error}")
            results["volatility"] = vol_result.data
            
            # Additional dates if requested
            if additional_dates:
                results["volatility_trend"] = []
                for date_str in additional_dates:
                    target_date = date.fromisoformat(date_str)
                    res = self._volatility.execute(target_date=target_date)
                    if res.success:
                        results["volatility_trend"].append(res.data)
        
        # Generate risk interpretation
        interpretation = self._generate_risk_interpretation(results)
        
        return {
            "tool_used": tool_name,
            "parameters": parameters,
            "results": results,
            "interpretation": interpretation,
            "risk_focus": decision["risk_interpretation_focus"]
        }
    
    def _generate_risk_interpretation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate risk interpretation from model predictions
        
        Args:
            results: Prediction results
            
        Returns:
            Dict with interpretation insights
        """
        interpretation = {
            "summary": "",
            "key_findings": [],
            "risk_alerts": []
        }
        
        # Crisis analysis
        if "crisis" in results:
            crisis = results["crisis"]
            prob = crisis.probability
            risk_level = crisis.risk_level
            
            interpretation["key_findings"].append(
                f"Crisis probability: {prob:.1%} ({risk_level} risk)"
            )
            
            # Alert on high risk
            if prob > 0.5:
                interpretation["risk_alerts"].append(
                    f"HIGH CRISIS RISK: {prob:.1%} probability within next 10 days"
                )
            
            # Trend analysis
            if "crisis_trend" in results:
                trend = results["crisis_trend"]
                if len(trend) >= 2:
                    prob_change = trend[-1].probability - trend[0].probability
                    if prob_change > 0.1:
                        interpretation["risk_alerts"].append(
                            f"Crisis risk increasing: +{prob_change:.1%} over period"
                        )
                    elif prob_change < -0.1:
                        interpretation["key_findings"].append(
                            f"Crisis risk decreasing: {prob_change:.1%} over period"
                        )
        
        # Volatility analysis
        if "volatility" in results:
            vol = results["volatility"]
            vol_value = vol.predicted_volatility
            regime = vol.volatility_regime
            
            interpretation["key_findings"].append(
                f"Predicted volatility: {vol_value:.1%} ({regime} regime)"
            )
            
            # Alert on extreme volatility
            if vol_value > 0.40:
                interpretation["risk_alerts"].append(
                    f"EXTREME VOLATILITY: {vol_value:.1%} (crisis-level market stress)"
                )
            elif vol_value > 0.25:
                interpretation["risk_alerts"].append(
                    f"HIGH VOLATILITY: {vol_value:.1%} (elevated market stress)"
                )
            
            # Trend analysis
            if "volatility_trend" in results:
                trend = results["volatility_trend"]
                if len(trend) >= 2:
                    vol_change = trend[-1].predicted_volatility - trend[0].predicted_volatility
                    if vol_change > 0.05:
                        interpretation["risk_alerts"].append(
                            f"Volatility rising: +{vol_change:.1%} over period"
                        )
        
        # Create summary
        num_alerts = len(interpretation["risk_alerts"])
        if num_alerts > 0:
            interpretation["summary"] = f"{num_alerts} risk alert(s) detected"
        else:
            interpretation["summary"] = "Risk levels within normal ranges"
        
        return interpretation