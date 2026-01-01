import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Stochastic Causal GraphRAG Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A4A5A;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #1E3A5F;
    }
    .assistant-message {
        background-color: #2D3748;
    }
    .agent-log {
        font-family: monospace;
        font-size: 0.85rem;
        padding: 0.5rem;
        background-color: #1A1A2E;
        border-radius: 4px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)


#INITIALIZATION

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent_logs' not in st.session_state:
        st.session_state.agent_logs = []
    if 'current_conversation_id' not in st.session_state:
        st.session_state.current_conversation_id = None
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'memory' not in st.session_state:
        st.session_state.memory = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'conversation_rag' not in st.session_state:
        st.session_state.conversation_rag = None


def load_data():
    """Load parquet data for statistics dashboard"""
    if st.session_state.data is None:
        try:
            data_path = Path("data/stochastic_market_data.parquet")
            if data_path.exists():
                st.session_state.data = pd.read_parquet(data_path)
                st.session_state.data['Date'] = pd.to_datetime(st.session_state.data['Date'])
        except Exception as e:
            st.error(f"Failed to load data: {e}")
    return st.session_state.data


def init_agents():
    """Initialize agent workflow (lazy loading)"""
    if st.session_state.workflow is None:
        try:
            from agents.agent_workflow import AgentWorkflow
            st.session_state.workflow = AgentWorkflow()
        except Exception as e:
            st.error(f"Failed to initialize agents: {e}")
    return st.session_state.workflow


def init_memory():
    """Initialize chat memory (lazy loading)"""
    if st.session_state.memory is None:
        try:
            from memory.chat_memory import ChatMemory
            st.session_state.memory = ChatMemory()
        except Exception as e:
            st.warning(f"Memory system unavailable: {e}")
    return st.session_state.memory


def init_conversation_rag():
    """Initialize conversation RAG for semantic search (lazy loading)"""
    if st.session_state.conversation_rag is None:
        try:
            from memory.conversation_rag import ConversationRAG
            st.session_state.conversation_rag = ConversationRAG()
        except Exception as e:
            st.warning(f"Conversation RAG unavailable: {e}")
    return st.session_state.conversation_rag


def store_message_in_rag(role: str, content: str):
    """Store message in RAG for future retrieval"""
    rag = init_conversation_rag()
    if rag:
        try:
            conv_id = st.session_state.current_conversation_id or "default"
            rag.add_message(conv_id, role, content)
        except Exception as e:
            pass 


#CHATBOT

def render_chatbot_tier():
    """Render the main chatbot interface"""
    st.header("💬 Financial Market Analysis Chatbot")
    
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
                    
                    # Show visualizations if any
                    if "visualizations" in message and message["visualizations"]:
                        for viz_path in message["visualizations"]:
                            if Path(viz_path).exists():
                                st.image(viz_path)
    
    # Chat input
    if prompt := st.chat_input("Ask about financial market data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with agents
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response, agent_logs, visualizations, comm_logs = process_query(prompt)
                st.markdown(response)
                
                # Show visualizations
                for viz_path in visualizations:
                    if Path(viz_path).exists():
                        st.image(viz_path)
        
        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "visualizations": visualizations
        })
        
        # Store agent logs
        st.session_state.agent_logs.extend(agent_logs)
        
        # Save to memory
        memory = init_memory()
        if memory:
            try:
                if st.session_state.current_conversation_id is None:
                    st.session_state.current_conversation_id = memory.new_conversation(prompt)
                memory.add_user_message(st.session_state.current_conversation_id, prompt)
                memory.add_assistant_message(
                    st.session_state.current_conversation_id, 
                    response,
                    visualizations=visualizations
                )
            except Exception as e:
                st.warning(f"Failed to save to memory: {e}")
        
        # Store in RAG for semantic search
        store_message_in_rag("user", prompt)
        store_message_in_rag("assistant", response)


def process_query(query: str) -> tuple[str, List[Dict], List[str], List[Dict]]:
    """
    Process user query through agent workflow
    
    Returns:
        Tuple of (response_text, agent_logs, visualization_paths, communication_logs)
    """
    agent_logs = []
    visualizations = []
    communication_logs = []
    
    try:
        workflow = init_agents()
        if workflow is None:
            return "Agent system not available. Please check configuration.", [], [], []
        
        # Execute query - now returns AgentWorkflowResult
        result = workflow.execute_query(query)
        report = result.report
        
        # Convert communication logs to dicts for session state
        communication_logs = [log.model_dump() for log in result.communication_logs]
        
        # Collect agent logs (legacy format for backward compatibility)
        agent_logs.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "agents_used": report.agents_used,
            "confidence": report.confidence_score,
            "findings_count": len(report.findings),
            "communication_logs": communication_logs
        })
        
        # Collect visualizations
        visualizations = report.visualizations
        
        # Format response
        response = f"## Summary\n{report.summary}\n\n"
        
        if report.findings:
            response += "## Detailed Findings\n"
            for finding in report.findings:
                category = finding.get("category", "General")
                response += f"\n### {category}\n"
                for point in finding.get("key_points", []):
                    response += f"- {point}\n"
        
        if report.recommendations:
            response += "\n## Recommendations\n"
            for i, rec in enumerate(report.recommendations, 1):
                response += f"{i}. {rec}\n"
        
        response += f"\n---\n*Confidence: {report.confidence_score:.0%} | Agents: {', '.join(report.agents_used)}*"
        
        return response, agent_logs, visualizations, communication_logs
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        agent_logs.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "error": str(e)
        })
        return error_msg, agent_logs, [], []


#AGENT COMMUNICATION

def render_agent_communication_tier():
    """Render agent communication timeline panel"""
    st.header("🤖 Agent Communication Panel")
    
    if not st.session_state.agent_logs:
        st.info("No agent activity yet. Start a conversation in the Chatbot tab.")
        return
    
    # Filter options
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        show_errors_only = st.checkbox("Show Errors Only")
    with col2:
        show_metadata = st.checkbox("Show Details", value=True)
    
    # Display logs for each query
    for i, log in enumerate(reversed(st.session_state.agent_logs)):
        if show_errors_only and "error" not in log:
            continue
        
        timestamp = log.get("timestamp", "Unknown time")
        query = log.get("query", "")[:80]
        
        with st.expander(f"🔹 Query: {query}...", expanded=(i == 0)):
            # Summary metrics
            if "error" not in log:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    agents = log.get("agents_used", [])
                    st.metric("Agents", len(agents))
                with col2:
                    confidence = log.get("confidence", 0)
                    st.metric("Confidence", f"{confidence:.0%}")
                with col3:
                    findings = log.get("findings_count", 0)
                    st.metric("Findings", findings)
                with col4:
                    comm_logs = log.get("communication_logs", [])
                    st.metric("Steps", len(comm_logs))
                
                st.divider()
            
            # Communication timeline
            comm_logs = log.get("communication_logs", [])
            if comm_logs:
                render_communication_timeline(comm_logs, show_metadata)
            elif "error" in log:
                st.error(f"❌ Error: {log['error']}")
            else:
                st.warning("No detailed communication logs available for this query.")
    
    # Clear logs button
    st.divider()
    if st.button("🗑️ Clear All Logs"):
        st.session_state.agent_logs = []
        st.rerun()


def render_communication_timeline(comm_logs: List[Dict], show_metadata: bool):
    """Render communication logs as a visual timeline"""
    
    # Define icons and colors for each step type
    step_styles = {
        "user_input": {"icon": "👤", "color": "#1E3A5F", "label": "User Input"},
        "orchestrator_plan": {"icon": "🧠", "color": "#4A4A8A", "label": "Orchestrator Plan"},
        "agent_task": {"icon": "📋", "color": "#2D5A3A", "label": "Task Assigned"},
        "agent_response": {"icon": "✅", "color": "#1F5A3A", "label": "Agent Response"},
        "tool_call": {"icon": "🔧", "color": "#5A4A2D", "label": "Tool Call"},
        "error": {"icon": "❌", "color": "#5A2D2D", "label": "Error"},
        "report": {"icon": "📊", "color": "#3A4A5A", "label": "Final Report"},
    }
    
    for idx, log_entry in enumerate(comm_logs):
        step_type = log_entry.get("step_type", "unknown")
        style = step_styles.get(step_type, {"icon": "•", "color": "#333", "label": step_type})
        agent_type = log_entry.get("agent_type", "")
        content = log_entry.get("content", "")
        metadata = log_entry.get("metadata", {})
        
        # Create visual timeline element
        agent_badge = f" [{agent_type.upper()}]" if agent_type else ""
        
        st.markdown(f"""
        <div style="
            border-left: 3px solid {style['color']};
            padding-left: 15px;
            margin-left: 10px;
            margin-bottom: 10px;
        ">
            <div style="
                background-color: {style['color']}33;
                padding: 10px 15px;
                border-radius: 0 8px 8px 0;
            ">
                <strong>{style['icon']} {style['label']}{agent_badge}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Content container
        with st.container():
            # Main content
            if step_type == "user_input":
                st.markdown(f"**Query:** {content}")
                
            elif step_type == "orchestrator_plan":
                st.markdown(f"**{content}**")
                if show_metadata and metadata:
                    tasks = metadata.get("tasks", [])
                    if tasks:
                        st.markdown("**Planned Tasks:**")
                        for t in tasks:
                            st.markdown(f"- `{t.get('agent', 'unknown')}`: {t.get('instruction', '')[:100]}...")
                    strategy = metadata.get("execution_strategy", "")
                    if strategy:
                        st.caption(f"Strategy: {strategy}")
                        
            elif step_type == "agent_task":
                st.markdown(f"**Instruction:** {content}")
                if show_metadata and metadata:
                    task_id = metadata.get("task_id", "")
                    if task_id:
                        st.caption(f"Task ID: `{task_id}`")
                        
            elif step_type == "agent_response":
                st.success(content)
                if show_metadata and metadata:
                    cols = st.columns(3)
                    with cols[0]:
                        tools = metadata.get("tools_used", [])
                        if tools:
                            st.markdown(f"**Tools:** {', '.join(tools)}")
                    with cols[1]:
                        iters = metadata.get("iterations", 0)
                        st.markdown(f"**Iterations:** {iters}")
                    with cols[2]:
                        result_summary = metadata.get("result_summary", "")
                        if result_summary:
                            st.markdown(f"**Result:** {result_summary[:50]}...")
                            
            elif step_type == "error":
                st.error(f"**Error:** {content}")
                if show_metadata and metadata:
                    task_id = metadata.get("task_id", "")
                    if task_id:
                        st.caption(f"Failed Task: `{task_id}`")
                        
            elif step_type == "report":
                st.markdown(f"**Summary:** {content}")
                if show_metadata and metadata:
                    cols = st.columns(3)
                    with cols[0]:
                        conf = metadata.get("confidence_score", 0)
                        st.metric("Confidence", f"{conf:.0%}")
                    with cols[1]:
                        findings = metadata.get("findings_count", 0)
                        st.metric("Findings", findings)
                    with cols[2]:
                        viz = metadata.get("visualizations_count", 0)
                        st.metric("Visualizations", viz)
                    
                    recs = metadata.get("recommendations", [])
                    if recs:
                        with st.expander("� Recommendations"):
                            for r in recs:
                                st.markdown(f"- {r}")
            else:
                st.text(content)
        
        # Add spacing between timeline items
        st.markdown("<br>", unsafe_allow_html=True)


#STATISTICS DASHBOARD

def render_statistics_tier():
    """Render Plotly statistics dashboard"""
    st.header("📊 Data Statistics Dashboard")
    
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check data/stochastic_market_data.parquet exists.")
        return
    
    # Dashboard tabs
    stat_tab1, stat_tab2, stat_tab3, stat_tab4 = st.tabs([
        "📈 Asset Prices", 
        "🏛️ Macro Indicators", 
        "🔥 Regime Analysis",
        "🔗 Correlations"
    ])
    
    with stat_tab1:
        render_asset_prices(df)
    
    with stat_tab2:
        render_macro_indicators(df)
    
    with stat_tab3:
        render_regime_analysis(df)
    
    with stat_tab4:
        render_correlations(df)


def render_asset_prices(df: pd.DataFrame):
    """Render asset price charts"""
    st.subheader("Asset Price Overview")
    
    # Asset selector
    asset_cols = [c for c in df.columns if '_Close' in c]
    selected_assets = st.multiselect(
        "Select Assets",
        asset_cols,
        default=asset_cols[:3]
    )
    
    if selected_assets:
        # Line chart
        fig = px.line(
            df, 
            x='Date', 
            y=selected_assets,
            title="Asset Closing Prices Over Time"
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Asset",
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Statistics table
        st.subheader("Summary Statistics")
        stats = df[selected_assets].describe().T
        st.dataframe(stats.style.format("{:.2f}"))


def render_macro_indicators(df: pd.DataFrame):
    """Render macroeconomic indicator charts"""
    st.subheader("Macroeconomic Indicators")
    
    macro_cols = ['GDP_Growth', 'Interest_Rate', 'Unemployment', 'Logistics', 'Production']
    available_cols = [c for c in macro_cols if c in df.columns]
    
    if not available_cols:
        st.warning("No macroeconomic columns found in data.")
        return
    
    selected_macro = st.selectbox("Select Indicator", available_cols)
    
    # Time series
    fig = px.line(
        df,
        x='Date',
        y=selected_macro,
        title=f"{selected_macro} Over Time"
    )
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, width='stretch')
    
    # Distribution
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            df, 
            x=selected_macro, 
            nbins=50,
            title=f"Distribution of {selected_macro}"
        )
        st.plotly_chart(fig_hist, width='stretch')
    
    with col2:
        fig_box = px.box(
            df, 
            y=selected_macro,
            title=f"Boxplot of {selected_macro}"
        )
        st.plotly_chart(fig_box, width='stretch')


def render_regime_analysis(df: pd.DataFrame):
    """Render regime analysis charts"""
    st.subheader("Market Regime Analysis")
    
    if 'Regime' not in df.columns:
        st.warning("Regime column not found in data.")
        return
    
    # Regime distribution
    regime_counts = df['Regime'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=regime_counts.values,
            names=regime_counts.index,
            title="Regime Distribution"
        )
        st.plotly_chart(fig_pie, width='stretch')
    
    with col2:
        # Regime timeline
        df_regime = df.copy()
        df_regime['Regime_Numeric'] = df_regime['Regime'].astype('category').cat.codes
        
        fig_timeline = px.scatter(
            df_regime,
            x='Date',
            y='Regime_Numeric',
            color='Regime',
            title="Regime Timeline"
        )
        fig_timeline.update_traces(marker=dict(size=3))
        st.plotly_chart(fig_timeline, width='stretch')
    
    # Statistics by regime
    st.subheader("Asset Performance by Regime")
    
    # Select asset
    asset_cols = [c for c in df.columns if '_Close' in c]
    if asset_cols:
        selected_asset = st.selectbox("Select Asset for Regime Analysis", asset_cols[:5])
        
        fig_violin = px.violin(
            df,
            x='Regime',
            y=selected_asset,
            box=True,
            title=f"{selected_asset} Distribution by Regime"
        )
        st.plotly_chart(fig_violin, width='stretch')


def render_correlations(df: pd.DataFrame):
    """Render correlation analysis"""
    st.subheader("Correlation Analysis")
    
    # Select columns for correlation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Remove Date-related if present
    cols_to_correlate = [c for c in numeric_cols if 'Date' not in c and 'Year' not in c][:15]
    
    selected_cols = st.multiselect(
        "Select Columns for Correlation Matrix",
        cols_to_correlate,
        default=cols_to_correlate[:6]
    )
    
    if len(selected_cols) >= 2:
        corr_matrix = df[selected_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Select at least 2 columns to show correlation matrix.")


#SCHEMA GRAPH

def render_schema_tier():
    """Render Neo4j schema visualization with Graphviz"""
    st.header("🕸️ Neo4j Schema Graph")
    
    # Check if graphviz is available
    try:
        import graphviz
    except ImportError:
        st.error("Graphviz not installed. Install with: pip install graphviz")
        st.info("You also need graphviz system package: apt install graphviz")
        return
    
    # Filter options
    st.sidebar.subheader("Schema Filters")
    
    node_types = st.sidebar.multiselect(
        "Node Types",
        ["Event", "Asset", "MacroVariable"],
        default=["Event", "Asset", "MacroVariable"]
    )
    
    rel_types = st.sidebar.multiselect(
        "Relationship Types",
        ["AFFECTS", "CO_MOVES_WITH", "SENSITIVE_TO"],
        default=["AFFECTS", "CO_MOVES_WITH", "SENSITIVE_TO"]
    )
    
    # Generate schema graph
    graph = generate_schema_graph(node_types, rel_types)
    
    if graph:
        st.graphviz_chart(graph, width='stretch')
        
        # Schema details
        with st.expander("📋 Schema Details"):
            render_schema_details()
    else:
        st.warning("No schema elements selected.")


def generate_schema_graph(node_types: List[str], rel_types: List[str]):
    """Generate Graphviz schema diagram"""
    try:
        import graphviz
        
        dot = graphviz.Digraph(comment='Neo4j Schema')
        dot.attr(rankdir='LR', bgcolor='transparent')
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        
        # Node colors
        colors = {
            'Event': '#FF6B6B',
            'Asset': '#4ECDC4',
            'MacroVariable': '#45B7D1'
        }
        
        # Add nodes
        for node_type in node_types:
            dot.node(
                node_type, 
                node_type,
                fillcolor=colors.get(node_type, '#CCCCCC'),
                fontcolor='white'
            )
        
        # Add relationships
        relationships = [
            ('Event', 'Asset', 'AFFECTS'),
            ('Asset', 'Asset', 'CO_MOVES_WITH'),
            ('Asset', 'MacroVariable', 'SENSITIVE_TO')
        ]
        
        for src, tgt, rel in relationships:
            if rel in rel_types and src in node_types and tgt in node_types:
                dot.edge(src, tgt, label=rel, fontname='Arial', fontsize='10')
        
        return dot
        
    except Exception as e:
        st.error(f"Error generating schema: {e}")
        return None


def render_schema_details():
    """Render detailed schema information"""
    st.markdown("""
    ### Node Types
    
    | Node | Properties |
    |------|------------|
    | **Event** | id, headline, body, type, date, affected_sector, regime, embedding |
    | **Asset** | id, sector, name |
    | **MacroVariable** | id, name, unit |
    
    ### Relationship Types
    
    | Relationship | From → To | Properties |
    |--------------|-----------|------------|
    | **AFFECTS** | Event → Asset | weight (-1 to 1), causal_lag (days) |
    | **CO_MOVES_WITH** | Asset → Asset | correlation (-1 to 1), direction |
    | **SENSITIVE_TO** | Asset → MacroVariable | beta (coefficient), factor |
    
    ### Sector Codes
    - TEC: Technology
    - IND: Industrial
    - FIN: Financial
    - ENE: Energy
    - HEA: Healthcare
    """)


#SIDEBAR

def render_sidebar():
    """Render sidebar with conversation history and settings"""
    st.sidebar.title("📁 Conversations")
    
    # New conversation button
    if st.sidebar.button("➕ New Conversation", width='stretch'):
        st.session_state.messages = []
        st.session_state.agent_logs = []
        st.session_state.current_conversation_id = None
        st.rerun()
    
    st.sidebar.divider()
    
    # List past conversations
    memory = init_memory()
    if memory:
        conversations = memory.list_conversations()
        
        if conversations:
            st.sidebar.subheader("History")
            for conv in conversations[:10]:  # Show last 10
                conv_id = conv.get("id", "")
                title = conv.get("title", "Untitled")[:30]
                msg_count = conv.get("message_count", 0)
                
                col1, col2 = st.sidebar.columns([4, 1])
                with col1:
                    if st.button(f"📝 {title}", key=f"conv_{conv_id}"):
                        load_conversation(conv_id)
                with col2:
                    if st.button("🗑️", key=f"del_{conv_id}"):
                        memory.delete_conversation(conv_id)
                        st.rerun()
    
    st.sidebar.divider()
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    
    st.sidebar.info("""
    **Stochastic Causal GraphRAG Agent**
    
    A multi-agent system for financial market analysis with:
    - Graph Agent (Neo4j)
    - Causal Agent (Tigramite)
    - Statistical Agent
    - Risk Agent (ML models)
    """)


def load_conversation(conversation_id: str):
    """Load a conversation from memory"""
    memory = init_memory()
    if memory:
        try:
            messages = memory.get_history(conversation_id)
            st.session_state.messages = [
                {"role": msg["role"], "content": msg["content"], "visualizations": msg.get("visualizations", [])}
                for msg in messages
            ]
            st.session_state.current_conversation_id = conversation_id
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load conversation: {e}")


#MAIN

def main():
    """Main application entry point"""
    init_session_state()
    
    # Sidebar
    render_sidebar()
    
    # Main content with 4 tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chatbot",
        "🤖 Agent Communication", 
        "📊 Statistics",
        "🕸️ Schema Graph"
    ])
    
    with tab1:
        render_chatbot_tier()
    
    with tab2:
        render_agent_communication_tier()
    
    with tab3:
        render_statistics_tier()
    
    with tab4:
        render_schema_tier()


if __name__ == "__main__":
    main()