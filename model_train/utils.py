import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
DATA_PATH = 'data/stochastic_market_data.parquet'
MODELS_DIR = 'models'
FORECAST_HORIZON = 10
CRISIS_THRESHOLD = -0.10
VOLATILITY_WINDOW = 10


def create_directories() -> None:
    """Create model storage directory structure."""
    sub_dirs = ['XGBoost', 'LSTM', 'EconML', 'Tigramite']
    for sub in sub_dirs:
        path = os.path.join(MODELS_DIR, sub)
        os.makedirs(path, exist_ok=True)
    logger.info("'%s' directory structure ready.", MODELS_DIR)


def _find_price_column(df: pd.DataFrame) -> str:
    """Find the primary price column in the dataframe."""
    price_col = 'Asset_01_TEC_Close'
    if price_col not in df.columns:
        potential = [c for c in df.columns if 'Close' in c]
        price_col = potential[0] if potential else None
    if not price_col:
        raise ValueError("Price column not found!")
    return price_col


def _calculate_future_targets(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Calculate forward-looking target variables for model training."""
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=FORECAST_HORIZON)
    
    # Crisis Target: Max Drawdown < -10% in next N days
    future_low = df[price_col].rolling(window=indexer).min()
    df['Future_MDD'] = (future_low - df[price_col]) / df[price_col]
    
    is_shock = df['Shock_Active'] == True if 'Shock_Active' in df.columns else False
    df['Target_Crisis'] = ((df['Future_MDD'] < CRISIS_THRESHOLD) | is_shock).astype(int)
    
    # Volatility Target: Future realized volatility (annualized)
    future_std = df['Log_Ret'].rolling(window=indexer).std()
    df['Target_Volatility'] = future_std * np.sqrt(252)
    
    return df


def load_and_preprocess_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load and preprocess market data with target calculations.
    
    Args:
        filepath: Path to the parquet data file.
        
    Returns:
        Preprocessed DataFrame with target columns.
    """
    df = pd.read_parquet(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Numeric conversion
    cols_check = ['Logistics', 'Production', 'Interest_Rate', 'Target_Rate']
    for c in cols_check:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    price_col = _find_price_column(df)
    df['Log_Ret'] = np.log(df[price_col] / df[price_col].shift(1))
    
    df = _calculate_future_targets(df, price_col)
    df = df.dropna(subset=['Future_MDD', 'Target_Volatility', 'Log_Ret']).reset_index(drop=True)

    return df


def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))


def _calculate_macd(prices: pd.Series) -> pd.Series:
    """Calculate Moving Average Convergence Divergence."""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    return exp1 - exp2


def _calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> tuple:
    """
    Calculate Bollinger Bands.
    
    Returns:
        Tuple of (upper_band, lower_band, percent_b)
    """
    bb_mean = prices.rolling(window).mean()
    bb_std = prices.rolling(window).std()
    
    upper = bb_mean + (bb_std * num_std)
    lower = bb_mean - (bb_std * num_std)
    percent_b = (prices - lower) / (upper - lower + 1e-6)
    
    return upper, lower, percent_b


def _add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add macroeconomic indicator features."""
    if 'Logistics' in df.columns and 'Production' in df.columns:
        df['Macro_SupplyChain_Stress'] = df['Logistics'] / (df['Production'] + 1e-6)
    
    if 'Interest_Rate' in df.columns and 'Target_Rate' in df.columns:
        df['Macro_Rate_Gap'] = df['Interest_Rate'] - df['Target_Rate']
        df['Macro_Rate_ROC'] = df['Interest_Rate'].diff()
    
    return df


def _add_technical_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Add technical analysis indicator features."""
    prices = df[price_col]
    
    df['Tech_RSI'] = _calculate_rsi(prices)
    df['Tech_MACD'] = _calculate_macd(prices)
    
    ma_50 = prices.rolling(window=50).mean()
    df['Tech_Dist_MA50'] = (prices - ma_50) / ma_50
    
    upper, lower, percent_b = _calculate_bollinger_bands(prices)
    df['Tech_BB_Upper'] = upper
    df['Tech_BB_Lower'] = lower
    df['Tech_BB_Pos'] = percent_b
    
    return df


def feature_engineering_common(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering with no data leakage.
    Uses only t and t-n values (no future information).
    
    Args:
        df: Input DataFrame with raw data.
        
    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()
    price_col = _find_price_column(df)
    
    df = _add_macro_features(df)
    df = _add_technical_features(df, price_col)
    
    # Regime encoding
    if 'Regime' in df.columns:
        dummies = pd.get_dummies(df['Regime'], prefix='Regime').astype(int)
        df = pd.concat([df, dummies], axis=1)

    # Drop NaN rows from feature calculations
    calc_features = ['Tech_RSI', 'Tech_Dist_MA50', 'Tech_BB_Pos']
    df = df.dropna(subset=[c for c in calc_features if c in df.columns]).reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Select numeric columns suitable for model training."""
    exclude_cols = [
        'Date', 'Year', 'News_Headline', 'News_Body', 'Event_Type', 
        'Event_ID', 'Affected_Sector', 'Shock_Active', 'Regime',
        'Future_MDD', 'Target_Crisis', 'Target_Volatility', 'Log_Ret'
    ]
    
    feats = [c for c in df.columns if c not in exclude_cols]
    return [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]


def calculate_log_returns(series: pd.Series) -> pd.Series:
    """Convert price series to log returns: ln(Pt / Pt-1)."""
    return np.log(series / series.shift(1))


def calculate_difference(series: pd.Series) -> pd.Series:
    """Apply first difference transformation to a series."""
    return series.diff()


def prepare_data_for_causal_discovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for causal discovery (Tigramite, EconML).
    Transforms data to I(0) stationary format.
    
    Args:
        df: Input DataFrame with raw data.
        
    Returns:
        DataFrame with stationary transformations applied.
    """
    df_causal = df.copy()
    
    # Transform prices to log returns
    price_cols = [c for c in df.columns if 'Close' in c]
    for col in price_cols:
        new_col = col.replace('_Close', '_LogRet')
        df_causal[new_col] = calculate_log_returns(df[col])
    
    # Transform macro variables with first difference
    macro_cols = ['Interest_Rate', 'Unemployment', 'GDP_Growth', 'Logistics', 'Production']
    for col in macro_cols:
        if col in df.columns:
            df_causal[f"{col}_Diff"] = calculate_difference(df[col])

    # Regime and Shock encoding
    if 'Regime' in df.columns:
        df_causal['Regime_Code'] = df['Regime'].astype('category').cat.codes
        dummies = pd.get_dummies(df['Regime'], prefix='Regime', drop_first=False)
        df_causal = pd.concat([df_causal, dummies], axis=1)
        
    if 'Shock_Active' in df.columns:
        df_causal['Shock_Active'] = df_causal['Shock_Active'].astype(int)

    df_causal = df_causal.dropna().reset_index(drop=True)
    
    return df_causal