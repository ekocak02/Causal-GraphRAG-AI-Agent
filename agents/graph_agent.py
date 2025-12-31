from typing import List, Dict, Any, Optional

from agents.base_agent import BaseAgent
from agents.tools.graph_tools import SemanticSearchTool, CypherQueryTool, get_neo4j_schema
from agents.prompts.system_prompts import GRAPH_AGENT_SYSTEM_PROMPT
from agents.config import GRAPH_MODEL


class GraphAgent(BaseAgent):
    """
    Specialist agent for Neo4j database queries
    
    Capabilities:
    - Semantic search on event embeddings
    - Custom Cypher queries for graph traversal
    """
    
    def __init__(self):
        super().__init__(model_name=GRAPH_MODEL)
        
        self._semantic_search = SemanticSearchTool()
        self._cypher_query = CypherQueryTool()
        
        # Cache Neo4j schema
        self.schema = get_neo4j_schema()
        
        # Dangerous Cypher keywords to block (must be standalone words)
        self._dangerous_patterns = [
            r'\bDELETE\b', r'\bREMOVE\b', r'\bDROP\b', r'\bDETACH\b', 
            r'\bSET\b', r'\bCREATE\b', r'\bMERGE\b', r'\bFOREACH\b'
        ]
    
    def validate_cypher(self, query: str) -> tuple[bool, str]:
        """
        Validate Cypher query for security and basic syntax
        
        Args:
            query: Cypher query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        import re
        query_upper = query.upper()
        
        # Check for dangerous operations using word boundaries
        for pattern in self._dangerous_patterns:
            if re.search(pattern, query_upper):
                keyword = pattern.replace(r'\b', '')
                return False, f"Security violation: {keyword} operation is not allowed"
        
        # Basic syntax checks
        if 'MATCH' not in query_upper and 'RETURN' not in query_upper:
            return False, "Query must contain MATCH and RETURN clauses"
        
        # Check for RETURN clause
        if 'RETURN' not in query_upper:
            return False, "Query must have a RETURN clause"
        
        return True, ""
    
    @property
    def agent_type(self) -> str:
        return "graph"
    
    @property
    def system_prompt(self) -> str:
        return GRAPH_AGENT_SYSTEM_PROMPT
    
    @property
    def available_tools(self) -> List[Any]:
        return [self._semantic_search, self._cypher_query]
    
    def _process_task(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process graph query task
        
        Args:
            instruction: Query instruction
            context: Additional context
            
        Returns:
            Query results
        """
        # Prepare prompt with schema and tools
        prompt = f"""
TASK: {instruction}

NEO4J SCHEMA:
- Node Labels: {', '.join(self.schema['node_labels'])}
- Relationship Types: {', '.join(self.schema['relationship_types'])}
- Node Properties: {self.schema['node_properties']}
- Relationship Properties: {self.schema['relationship_properties']}

AVAILABLE TOOLS:
{self.format_tool_descriptions()}

INSTRUCTIONS:
1. Decide which tool to use (semantic_search for text queries, cypher_query for relationships)
2. Provide the tool name and parameters
3. Explain your reasoning

OUTPUT as JSON:
{{
    "tool": "semantic_search|cypher_query",
    "parameters": {{
        // tool-specific parameters
    }},
    "reasoning": "brief explanation"
}}
"""
        
        # Get tool decision from LLM
        decision = self.call_llm(prompt, temperature=0.1, format="json")
        
        tool_name = decision["tool"]
        parameters = decision["parameters"]
        
        self.logger.info(f"Using tool: {tool_name}")
        self.logger.debug(f"Parameters: {parameters}")
        
        # Validate Cypher query if using cypher_query tool
        if tool_name == "cypher_query" and "query" in parameters:
            is_valid, error_msg = self.validate_cypher(parameters["query"])
            if not is_valid:
                self.logger.warning(f"Cypher validation failed: {error_msg}")
                raise ValueError(f"Cypher query validation failed: {error_msg}")
        
        # Execute tool
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        result = tool.execute(**parameters)
        
        if not result.success:
            raise RuntimeError(f"Tool execution failed: {result.error}")
        
        # Format result for downstream agents
        return {
            "tool_used": tool_name,
            "parameters": parameters,
            "result": result.data,
            "reasoning": decision["reasoning"]
        }