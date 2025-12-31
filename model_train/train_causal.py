import os
import json
import pickle
import logging
import numpy as np
from sklearn.tree import export_text

from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter
from xgboost import XGBRegressor, XGBClassifier

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.data_processing import DataFrame as TigDataFrame

from utils import (
    load_and_preprocess_data, 
    create_directories, 
    prepare_data_for_causal_discovery, 
    MODELS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_discovery_variables(df) -> list:
    """Auto-discover stationary variables for causal analysis."""
    valid_vars = [
        c for c in df.columns 
        if '_LogRet' in c or '_Diff' in c or c in ['Shock_Active', 'Regime_Code', 'Vol_Mult']
    ]
    logger.info("AUTO-DISCOVERY: %d variables included in analysis.", len(valid_vars))
    return valid_vars


def _get_heterogeneity_features(df) -> list:
    """Find features that may modify treatment effects."""
    het_feats = [c for c in df.columns if c.startswith('Regime_') and c != 'Regime_Code']
    
    extras = ['Vol_Mult', 'Macro_SupplyChain_Stress', 'Tech_RSI']
    for e in extras:
        if e in df.columns:
            het_feats.append(e)
            
    return list(set(het_feats))


def _extract_confounders(graph_edges: list, treatment: str, outcome: str) -> list:
    """Extract confounding variables from causal graph."""
    confounders = set()
    
    for edge in graph_edges:
        src, tgt = edge['source'], edge['target']
        
        if src in (treatment, outcome):
            continue
        if tgt in (treatment, outcome):
            confounders.add(src)
            
    return list(confounders)


def _parse_pcmci_results(results: dict, var_names: list, alpha: float = 0.05) -> list:
    """Parse PCMCI results into edge list."""
    edges = []
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    
    for j in range(len(var_names)):
        for i in range(len(var_names)):
            for tau in range(6):
                if i == j:
                    continue
                    
                pval = p_matrix[i, j, tau]
                if pval < alpha:
                    edge = {
                        "source": var_names[i],
                        "target": var_names[j],
                        "lag": tau,
                        "strength": float(val_matrix[i, j, tau]),
                        "p_value": float(pval)
                    }
                    edges.append(edge)
                    
                    if abs(edge['strength']) > 0.1:
                        logger.info("FOUND: %s(lag-%d) --> %s (Str: %.2f)", 
                                   var_names[i], tau, var_names[j], edge['strength'])
    
    return edges


def run_tigramite_discovery(df_causal) -> list:
    """
    Run Tigramite PCMCI causal discovery.
    
    Returns:
        List of causal edge dictionaries.
    """
    logger.info("TIGRAMITE STRUCTURAL LEARNING")
    
    valid_vars = _get_discovery_variables(df_causal)
    
    if len(valid_vars) < 2:
        logger.error("Not enough variables for Tigramite analysis.")
        return []

    data = df_causal[valid_vars].values
    dataframe = TigDataFrame(data, var_names=valid_vars)
    
    logger.info("Running PCMCI (Variables: %d, Tau_max: 5)", len(valid_vars))
    
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05)
    
    # Save raw results
    save_path = os.path.join(MODELS_DIR, 'Tigramite', 'causal_graph_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Parse and save edges
    graph_edges = _parse_pcmci_results(results, valid_vars)
    
    json_path = os.path.join(MODELS_DIR, 'Tigramite', 'causal_edges.json')
    with open(json_path, 'w') as f:
        json.dump(graph_edges, f, indent=4)
        
    logger.info("Tigramite completed. %d causal edges found.", len(graph_edges))
    return graph_edges


def _prepare_treatment_data(df, source: str, lag: int):
    """Prepare treatment variable with optional lag shifting."""
    df_temp = df.copy()
    treatment_col = source
    
    if lag > 0:
        treatment_col = f"{source}_lag_{lag}"
        df_temp[treatment_col] = df_temp[source].shift(lag)
        df_temp = df_temp.dropna().reset_index(drop=True)
    
    return df_temp, treatment_col


def _is_discrete_treatment(T: np.ndarray, source: str) -> bool:
    """Determine if treatment should be modeled as discrete."""
    unique_values = set(np.unique(T))
    is_binary = unique_values.issubset({0, 1, 0.0, 1.0})
    is_categorical = 'Shock' in source or 'Regime' in source
    return len(unique_values) <= 5 and (is_binary or is_categorical)


def _create_nuisance_models(is_discrete: bool):
    """Create nuisance models for CausalForestDML."""
    model_y = XGBRegressor(n_estimators=200, max_depth=3, verbosity=0)
    
    if is_discrete:
        model_t = XGBClassifier(n_estimators=200, max_depth=3, verbosity=0)
    else:
        model_t = XGBRegressor(n_estimators=200, max_depth=3, verbosity=0)
    
    return model_y, model_t


def _fit_causal_forest(Y, T, X, W, is_discrete: bool):
    """Fit CausalForestDML estimator."""
    model_y, model_t = _create_nuisance_models(is_discrete)
    
    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=is_discrete,
        n_estimators=100,
        random_state=42
    )
    est.fit(Y, T, X=X, W=W)
    return est


def _extract_policy_tree(est, X, X_cols: list, scenario_id: str, models_dir: str) -> str:
    """Extract and save policy tree interpretation."""
    try:
        intrp = SingleTreeCateInterpreter(
            include_model_uncertainty=True, 
            max_depth=2, 
            min_samples_leaf=10
        )
        intrp.interpret(est, X)
        
        policy_text = export_text(intrp.tree_model_, feature_names=X_cols)
        
        policy_path = os.path.join(models_dir, f"{scenario_id}_policy_tree.pkl")
        with open(policy_path, 'wb') as f:
            pickle.dump(intrp, f)
        
        return policy_text
        
    except Exception as e:
        return f"Tree extraction failed: {str(e)}"


def _process_single_scenario(df_causal, edge: dict, graph_edges: list, heterogeneity_cols: list, models_dir: str) -> dict:
    """Process a single causal scenario."""
    source, target, lag = edge['source'], edge['target'], edge['lag']
    scenario_id = f"{source}_lag{lag}_to_{target}"
    
    logger.info("Scenario: %s (Lag: %d) -> %s", source, lag, target)
    
    # Prepare data
    df_temp, treatment_col = _prepare_treatment_data(df_causal, source, lag)
    
    if len(df_temp) < 50:
        logger.warning("SKIP: Insufficient data after lag shift.")
        return None

    # Extract confounders
    confounders = _extract_confounders(graph_edges, source, target)
    confounders = [c for c in confounders if c in df_temp.columns]
    
    # Prepare arrays
    Y = df_temp[target].astype(float).values
    T = df_temp[treatment_col].astype(int if df_temp[treatment_col].dtype == bool else float).values
    
    X_cols = [c for c in heterogeneity_cols if c in df_temp.columns]
    if not X_cols:
        X_cols = [c for c in df_temp.columns if 'Regime' in c]
    
    X = df_temp[X_cols].astype(float).values
    W = df_temp[confounders].astype(float).values if confounders else None
    
    # Fit model
    is_discrete = _is_discrete_treatment(T, source)
    est = _fit_causal_forest(Y, T, X, W, is_discrete)
    
    # Save model
    model_path = os.path.join(models_dir, f"{scenario_id}_forest.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(est, f)
    
    # Calculate ATE
    ate_val = est.ate(X).mean()
    logger.info("ATE: %.4f", ate_val)
    
    # Extract policy tree
    policy_text = _extract_policy_tree(est, X, X_cols, scenario_id, models_dir)
    
    return {
        "source": source,
        "target": target,
        "lag": lag,
        "strength": float(ate_val),
        "confounders": confounders,
        "policy_tree": policy_text
    }


def run_econml_inference(df_causal, graph_edges: list) -> None:
    """
    Run EconML heterogeneous treatment effect analysis.
    
    Args:
        df_causal: Prepared causal dataframe.
        graph_edges: List of causal edges from Tigramite.
    """
    logger.info("ECONML CAUSAL INFERENCE")
    
    if not graph_edges:
        logger.warning("No causal edges to process.")
        return

    models_dir = os.path.join(MODELS_DIR, 'EconML', 'Models')
    os.makedirs(models_dir, exist_ok=True)
    logger.info("Model directory: %s", models_dir)
    
    heterogeneity_cols = _get_heterogeneity_features(df_causal)
    logger.info("Heterogeneity features: %s", heterogeneity_cols)

    processed = set()
    results = []

    for edge in graph_edges:
        scenario_id = f"{edge['source']}_lag{edge['lag']}_to_{edge['target']}"
        
        if scenario_id in processed:
            continue
        processed.add(scenario_id)

        try:
            result = _process_single_scenario(
                df_causal, edge, graph_edges, heterogeneity_cols, models_dir
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.error("Scenario failed (%s): %s", scenario_id, e)
            continue

    # Save summary
    summary_path = os.path.join(MODELS_DIR, 'EconML', 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info("EconML completed. %d scenarios saved: %s", len(results), summary_path)


def main():
    """Main entry point for causal training pipeline."""
    create_directories()
    
    logger.info("Loading data...")
    df_raw = load_and_preprocess_data()
    
    logger.info("Preparing data for causal analysis...")
    df_causal = prepare_data_for_causal_discovery(df_raw)
    
    # Run discovery
    graph_edges = run_tigramite_discovery(df_causal)
    
    # Run inference
    run_econml_inference(df_causal, graph_edges)


if __name__ == "__main__":
    main()