from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import date, datetime


#TOOL INPUTS

class SemanticSearchInput(BaseModel):
    """Semantic search tool input"""
    query: str = Field(..., description="Search query text")
    limit: int = Field(default=5, description="Maximum number of results")
    filter_type: Optional[str] = Field(None, description="Filter by event type")
    filter_sector: Optional[str] = Field(None, description="Filter by affected sector")
    filter_regime: Optional[str] = Field(None, description="Filter by regime")


class CypherQueryInput(BaseModel):
    """Cypher query tool input"""
    query: str = Field(..., description="Cypher query to execute")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")


class StatisticalSummaryInput(BaseModel):
    """Statistical summary tool input"""
    target_column: str = Field(..., description="Target column for analysis")
    start_date: Optional[date] = Field(None, description="Start date for filtering")
    end_date: Optional[date] = Field(None, description="End date for filtering")
    x_column: Optional[str] = Field(None, description="X-axis column for visualization")
    y_column: Optional[str] = Field(None, description="Y-axis column for visualization")
    plot_type: Optional[Literal["line", "scatter", "hist", "box"]] = Field(
        None, description="Plot type: line, scatter, hist, box"
    )


class CorrelationMapInput(BaseModel):
    """Correlation map tool input"""
    columns: List[str] = Field(..., description="Columns to include in correlation analysis")
    start_date: Optional[date] = Field(None, description="Start date for filtering")
    end_date: Optional[date] = Field(None, description="End date for filtering")
    method: Literal["pearson", "spearman"] = Field(
        default="pearson", description="Correlation method"
    )


class DataValidationInput(BaseModel):
    """Data validation tool input"""
    start_date: date = Field(..., description="Start date for validation period")
    end_date: date = Field(..., description="End date for validation period")


class TigramiteFilterInput(BaseModel):
    """Tigramite causal edges filter input"""
    source: Optional[str] = Field(None, description="Filter by source variable")
    target: Optional[str] = Field(None, description="Filter by target variable")
    min_strength: Optional[float] = Field(None, description="Minimum absolute strength")
    max_lag: Optional[int] = Field(None, description="Maximum lag to include")
    limit: Optional[int] = Field(None, description="Maximum number of results")


class EconMLFilterInput(BaseModel):
    """EconML inference filter input"""
    source: Optional[str] = Field(None, description="Filter by source variable")
    target: Optional[str] = Field(None, description="Filter by target variable")
    min_strength: Optional[float] = Field(None, description="Minimum absolute strength")
    max_lag: Optional[int] = Field(None, description="Maximum lag to include")
    limit: Optional[int] = Field(None, description="Maximum number of results")


class CrisisInput(BaseModel):
    """Crisis prediction tool input"""
    target_date: date = Field(..., description="Date to predict crisis probability")
    model_choice: Literal["auto", "early", "late"] = Field(
        default="auto", 
        description="Model selection: auto (based on date), early (first 5 years), late (last 5 years)"
    )


class VolatilityInput(BaseModel):
    """Volatility prediction tool input"""
    target_date: date = Field(..., description="Date to predict volatility")


#TOOL OUTPUTS

class ToolResult(BaseModel):
    """Generic tool result wrapper"""
    success: bool = Field(..., description="Whether tool execution was successful")
    data: Any = Field(..., description="Tool output data")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SemanticSearchResult(BaseModel):
    """Semantic search result item"""
    event_id: str
    headline: str
    body: str
    event_type: str
    date: date
    affected_sector: Optional[str]
    regime: Optional[str]
    similarity_score: float


class CypherQueryResult(BaseModel):
    """Cypher query result"""
    records: List[Dict[str, Any]] = Field(..., description="Query result records")
    count: int = Field(..., description="Number of records returned")


class TigramiteEdge(BaseModel):
    """Single Tigramite causal edge"""
    source: str
    target: str
    lag: int
    strength: float
    p_value: float


class EconMLInference(BaseModel):
    """Single EconML inference result"""
    source: str
    target: str
    lag: int
    strength: float
    confounders: List[str]
    policy_tree: str


class CrisisPrediction(BaseModel):
    """Crisis prediction result"""
    date: date
    probability: float
    model_used: str
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    confidence: float


class VolatilityPrediction(BaseModel):
    """Volatility prediction result"""
    date: date
    predicted_volatility: float
    volatility_regime: Literal["Low", "Normal", "High", "Extreme"]


#AGENT MESSAGES

class AgentTask(BaseModel):
    """Task assigned to an agent"""
    task_id: str = Field(..., description="Unique task identifier")
    agent_type: Literal["graph", "causal", "statistical", "risk", "report"] = Field(
        ..., description="Target agent type"
    )
    instruction: str = Field(..., description="Task instruction")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    priority: int = Field(default=1, description="Task priority (1=highest)")


class AgentResponse(BaseModel):
    """Response from an agent"""
    task_id: str = Field(..., description="Task identifier")
    agent_type: str = Field(..., description="Agent type")
    success: bool = Field(..., description="Whether task was successful")
    result: Any = Field(..., description="Agent result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    tools_used: List[str] = Field(default_factory=list, description="Tools used by agent")
    iterations: int = Field(default=1, description="Number of iterations taken")


class OrchestratorPlan(BaseModel):
    """Orchestrator's execution plan"""
    tasks: List[AgentTask] = Field(..., description="Ordered list of tasks")
    execution_strategy: Literal["sequential", "parallel", "adaptive"] = Field(
        default="sequential", description="Execution strategy"
    )
    estimated_duration: Optional[float] = Field(None, description="Estimated duration in seconds")


class FinalReport(BaseModel):
    """Final report from Report Agent"""
    user_query: str = Field(..., description="Original user query")
    summary: str = Field(..., description="Executive summary")
    findings: List[Dict[str, Any]] = Field(..., description="Detailed findings")
    visualizations: List[str] = Field(default_factory=list, description="Paths to generated plots")
    recommendations: Optional[List[str]] = Field(None, description="Recommendations")
    confidence_score: float = Field(..., description="Overall confidence in results (0-1)")
    agents_used: List[str] = Field(..., description="List of agents that contributed")


#NEO4J SCHEMA

class Neo4jSchema(BaseModel):
    """Neo4j database schema information"""
    node_labels: List[str] = Field(..., description="Available node labels")
    relationship_types: List[str] = Field(..., description="Available relationship types")
    node_properties: Dict[str, List[str]] = Field(..., description="Properties per node label")
    relationship_properties: Dict[str, List[str]] = Field(
        ..., description="Properties per relationship type"
    )


#ERROR HANDLING

class AgentError(BaseModel):
    """Agent error details"""
    error_type: Literal["tool_error", "validation_error", "llm_error", "timeout_error"]
    message: str
    agent_type: str
    task_id: Optional[str] = None
    retry_count: int = 0
    is_recoverable: bool = True


#COMMUNICATION LOGGING

class CommunicationLog(BaseModel):
    """Single communication step in the agent workflow"""
    step_type: Literal[
        "user_input",
        "orchestrator_plan", 
        "agent_task",
        "agent_response",
        "tool_call",
        "error",
        "report"
    ] = Field(..., description="Type of communication step")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this step occurred")
    agent_type: Optional[str] = Field(None, description="Agent involved (if applicable)")
    content: str = Field(..., description="Main content/message of this step")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional step-specific data")


class AgentWorkflowResult(BaseModel):
    """Complete workflow result including report and communication logs"""
    report: "FinalReport" = Field(..., description="Final analysis report")
    communication_logs: List[CommunicationLog] = Field(
        default_factory=list, 
        description="Ordered list of all communication steps"
    )