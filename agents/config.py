import os
from dotenv import load_dotenv

load_dotenv()

#OLLAMA MODEL CONFIGURATION

# Orchestrator Agent
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "mistral-nemo:12b")

# Graph Agent
GRAPH_MODEL = os.getenv("GRAPH_MODEL", "qwen2.5-coder:7b")

# Causal Agent
CAUSAL_MODEL = os.getenv("CAUSAL_MODEL", "qwen3:8b")

# Statistical Agent
STATISTICAL_MODEL = os.getenv("STATISTICAL_MODEL", "dolphin3:8b")

# Risk Agent
RISK_MODEL = os.getenv("RISK_MODEL", "dolphin3:8b")

# Report Agent
REPORT_MODEL = os.getenv("REPORT_MODEL", "mistral-nemo:12b")

# Embedding model for vector representations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")

#OLLAMA CONNECTION
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1000"))

#NEO4J CONNECTION
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

#FILE PATHS
DATA_PATH = "data/stochastic_market_data.parquet"
MODELS_DIR = "models"

TIGRAMITE_EDGES_PATH = "models/Tigramite/causal_edges.json"
ECONML_SUMMARY_PATH = "models/EconML/inference_summary.json"

XGB_EARLY_MODEL_PATH = "models/XGBoost/xgb_model_early.joblib"
XGB_LATE_MODEL_PATH = "models/XGBoost/xgb_model_late.joblib"
LSTM_MODEL_PATH = "models/LSTM/lstm_vol_final.pth"

#MODEL PARAMETERS
# LSTM parameters
LSTM_SEQ_LENGTH = 252
LSTM_HIDDEN_DIM = 64
LSTM_ATTENTION_DIM = 32

# XGBoost date threshold (5 years from start for early/late model selection)
XGB_DATE_THRESHOLD_YEARS = 5

#AGENT BEHAVIOR
MAX_TOOL_RETRIES = 2  
MAX_AGENT_ITERATIONS = 5  
ERROR_FEEDBACK_ENABLED = True  

#OUTPUT SETTINGS
ENABLE_PLOTTING = True 
FIGURE_DPI = 100
FIGURE_SIZE = (10, 6)

#LOGGING
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

#DATA SCHEMA
# Complete schema of stochastic_market_data.parquet (66 columns, 2520 rows)
# Date Range: 2024-01-01 to 2033-08-26 (10 years simulation)

DATA_SCHEMA = {
    "description": "10-year stochastic financial market simulation with OHLCV data for 10 assets, macroeconomic indicators, and market events",
    "total_columns": 66,
    "total_rows": 2520,
    "date_range": {"start": "2024-01-01", "end": "2033-08-26"},
    
    "columns": {
        # Date/Time columns
        "Date": {"type": "datetime", "description": "Trading date (daily frequency)"},
        "Year": {"type": "int", "description": "Simulation year (0-9)"},
        
        # Regime/Scenario columns
        "Regime": {"type": "str", "description": "Market regime: Growth, Shock, Recovery, Overheating, Stabilization"},
        "Target_Rate": {"type": "float", "description": "Central bank target interest rate"},
        "Vol_Mult": {"type": "float", "description": "Volatility multiplier for current regime"},
        "Shock_Active": {"type": "bool", "description": "Whether a shock event is currently active"},
        
        # Macroeconomic indicators
        "Logistics": {"type": "float", "description": "Logistics index (Lotka-Volterra model)"},
        "Production": {"type": "float", "description": "Production index (Lotka-Volterra model)"},
        "GDP_Growth": {"type": "float", "description": "GDP growth rate (annualized %)"},
        "Interest_Rate": {"type": "float", "description": "Current interest rate (Vasicek model)"},
        "Unemployment": {"type": "float", "description": "Unemployment rate (%)"},
        
        # Event/News columns
        "News_Headline": {"type": "str", "description": "Generated news headline for the day"},
        "News_Body": {"type": "str", "description": "Detailed news body text"},
        "Event_Type": {"type": "str", "description": "Event category: Policy, Crisis, Recovery, etc."},
        "Event_ID": {"type": "str", "description": "Unique event identifier"},
        "Affected_Sector": {"type": "str", "description": "Sector affected by event: TEC, IND, FIN, ENE, HLT, etc."},
    },
    
    # Asset OHLCV columns (10 assets × 5 columns = 50 columns)
    "assets": {
        "Asset_01_TEC": {"sector": "Technology", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_02_IND": {"sector": "Industrial", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_03_FIN": {"sector": "Financial", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_04_ENE": {"sector": "Energy", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_05_HLT": {"sector": "Healthcare", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_06_CST": {"sector": "Consumer Staples", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_07_CSD": {"sector": "Consumer Discretionary", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_08_UTL": {"sector": "Utilities", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_09_MAT": {"sector": "Materials", "columns": ["Close", "Open", "High", "Low", "Volume"]},
        "Asset_10_COM": {"sector": "Communications", "columns": ["Close", "Open", "High", "Low", "Volume"]},
    },
    
    # Column name patterns for easy lookup
    "column_patterns": {
        "price_columns": ["Asset_XX_YYY_Close", "Asset_XX_YYY_Open", "Asset_XX_YYY_High", "Asset_XX_YYY_Low"],
        "volume_columns": ["Asset_XX_YYY_Volume"],
        "macro_columns": ["GDP_Growth", "Interest_Rate", "Unemployment", "Logistics", "Production"],
        "regime_columns": ["Regime", "Target_Rate", "Vol_Mult", "Shock_Active"],
    },
    
    # Example column names for agent prompts
    "example_columns": [
        "Asset_01_TEC_Close",
        "Asset_02_IND_Close", 
        "Asset_03_FIN_Close",
        "GDP_Growth",
        "Interest_Rate",
        "Unemployment",
    ]
}

# Neo4j Graph Schema for Graph Agent
NEO4J_SCHEMA = {
    "nodes": {
        "Event": {
            "properties": {
                "id": "string - Unique event ID (e.g., EVT_2024_001)",
                "headline": "string - Short event headline",
                "body": "string - Detailed event description",
                "type": "string - Event type: Policy, Crisis, Recovery, Market, etc.",
                "date": "date - Event date",
                "affected_sector": "string - Affected sector code: TEC, IND, FIN, etc.",
                "regime": "string - Market regime when event occurred",
                "embedding": "vector - 768-dim embedding from embeddinggemma"
            },
            "example": "(:Event {id: 'EVT_2024_087', headline: 'Fed Raises Interest Rates', type: 'Policy'})"
        },
        "Asset": {
            "properties": {
                "id": "string - Asset identifier (e.g., Asset_01_TEC)",
                "sector": "string - Sector code: TEC, IND, FIN, ENE, HLT, CST, CSD, UTL, MAT, COM",
                "name": "string - Full sector name"
            },
            "example": "(:Asset {id: 'Asset_01_TEC', sector: 'TEC', name: 'Technology'})"
        },
        "MacroVariable": {
            "properties": {
                "id": "string - Variable identifier (e.g., GDP_Growth)",
                "name": "string - Full variable name",
                "unit": "string - Measurement unit (%, index, rate)"
            },
            "example": "(:MacroVariable {id: 'Interest_Rate', name: 'Interest Rate', unit: 'percent'})"
        }
    },
    "relationships": {
        "AFFECTS": {
            "from": "Event",
            "to": "Asset",
            "properties": {
                "weight": "float - Impact strength (-1 to 1)",
                "causal_lag": "int - Days until effect (0-30)"
            },
            "example": "(:Event)-[:AFFECTS {weight: -0.5, causal_lag: 3}]->(:Asset)"
        },
        "CO_MOVES_WITH": {
            "from": "Asset",
            "to": "Asset",
            "properties": {
                "correlation": "float - Pearson correlation (-1 to 1)",
                "direction": "string - positive/negative"
            },
            "example": "(:Asset)-[:CO_MOVES_WITH {correlation: 0.85, direction: 'positive'}]->(:Asset)"
        },
        "SENSITIVE_TO": {
            "from": "Asset",
            "to": "MacroVariable",
            "properties": {
                "beta": "float - Sensitivity coefficient",
                "factor": "string - Factor name"
            },
            "example": "(:Asset)-[:SENSITIVE_TO {beta: -0.7, factor: 'rate_sensitivity'}]->(:MacroVariable)"
        }
    }
}

# Cypher Query Examples for Graph Agent
CYPHER_EXAMPLES = [
    # Basic queries
    {
        "description": "Find all events affecting Technology sector",
        "query": "MATCH (e:Event)-[:AFFECTS]->(a:Asset {sector: 'TEC'}) RETURN e.headline, e.date, a.id LIMIT 10"
    },
    {
        "description": "Find assets sensitive to Interest Rate with negative beta",
        "query": "MATCH (a:Asset)-[r:SENSITIVE_TO]->(m:MacroVariable {id: 'Interest_Rate'}) WHERE r.beta < -0.5 RETURN a.id, a.sector, r.beta ORDER BY r.beta"
    },
    {
        "description": "Find highly correlated asset pairs",
        "query": "MATCH (a1:Asset)-[r:CO_MOVES_WITH]->(a2:Asset) WHERE r.correlation > 0.8 RETURN a1.id, a2.id, r.correlation ORDER BY r.correlation DESC LIMIT 10"
    },
    {
        "description": "Find events during Shock regime",
        "query": "MATCH (e:Event) WHERE e.regime = 'Shock' RETURN e.headline, e.date, e.type ORDER BY e.date"
    },
    {
        "description": "Find events with delayed impact (causal_lag > 5 days)",
        "query": "MATCH (e:Event)-[r:AFFECTS]->(a:Asset) WHERE r.causal_lag > 5 RETURN e.headline, a.id, r.causal_lag ORDER BY r.causal_lag DESC"
    },
    {
        "description": "Count events by type",
        "query": "MATCH (e:Event) RETURN e.type AS event_type, COUNT(*) AS count ORDER BY count DESC"
    },
    {
        "description": "Find assets affected by Policy events",
        "query": "MATCH (e:Event {type: 'Policy'})-[r:AFFECTS]->(a:Asset) RETURN DISTINCT a.id, a.sector, COUNT(e) AS policy_events ORDER BY policy_events DESC"
    },
    {
        "description": "Find negative correlations between sectors",
        "query": "MATCH (a1:Asset)-[r:CO_MOVES_WITH]->(a2:Asset) WHERE r.correlation < -0.3 AND a1.sector <> a2.sector RETURN a1.sector, a2.sector, r.correlation"
    },
    {
        "description": "Find recent events (last year of simulation)",
        "query": "MATCH (e:Event) WHERE e.date >= date('2033-01-01') RETURN e.headline, e.date, e.type ORDER BY e.date DESC LIMIT 10"
    },
    {
        "description": "Find path from event to macro variable through asset",
        "query": "MATCH path = (e:Event)-[:AFFECTS]->(a:Asset)-[:SENSITIVE_TO]->(m:MacroVariable) RETURN e.headline, a.id, m.id LIMIT 10"
    }
]