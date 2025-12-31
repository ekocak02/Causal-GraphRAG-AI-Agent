import pandas as pd
import joblib
import torch
import torch.nn as nn
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler

from agents.tools.base_tool import BaseTool
from agents.models.schemas import (
    CrisisInput,
    VolatilityInput,
    CrisisPrediction,
    VolatilityPrediction
)
from agents.config import (
    DATA_PATH,
    XGB_EARLY_MODEL_PATH,
    XGB_LATE_MODEL_PATH,
    LSTM_MODEL_PATH,
    XGB_DATE_THRESHOLD_YEARS,
    LSTM_SEQ_LENGTH,
    LSTM_HIDDEN_DIM,
    LSTM_ATTENTION_DIM
)


# LSTM Model Definition
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        attn_weights = self.attention(h_lstm)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * h_lstm, dim=1)
        output = self.fc(context)
        return output


class CrisisPredictionTool(BaseTool):
    """
    Predict crisis probability using XGBoost classifier
    """
    
    def __init__(self):
        super().__init__()
        self.df = pd.read_parquet(DATA_PATH)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Load both models
        self.model_early = joblib.load(XGB_EARLY_MODEL_PATH)
        self.model_late = joblib.load(XGB_LATE_MODEL_PATH)
        
        # Compute threshold date for model selection
        start_date = self.df['Date'].min()
        self.threshold_date = start_date + timedelta(days=365 * XGB_DATE_THRESHOLD_YEARS)
    
    @property
    def name(self) -> str:
        return "get_crisis"
    
    @property
    def description(self) -> str:
        return """Predict crisis probability for a target date using XGBoost classifier.
        Returns probability (0-1) and risk level (Low/Medium/High/Critical).
        Uses early model (years 0-5) or late model (years 5-10) automatically."""
    
    @property
    def input_schema(self) -> type[CrisisInput]:
        return CrisisInput
    
    def _execute(
        self,
        target_date: date,
        model_choice: str = "auto"
    ) -> CrisisPrediction:
        """
        Predict crisis probability
        
        Args:
            target_date: Date to predict
            model_choice: "auto", "early", or "late"
            
        Returns:
            CrisisPrediction with probability and risk level
        """
        target_dt = pd.to_datetime(target_date)
        
        # Select model
        if model_choice == "auto":
            model = self.model_late if target_dt >= self.threshold_date else self.model_early
            model_used = "late" if target_dt >= self.threshold_date else "early"
        elif model_choice == "early":
            model = self.model_early
            model_used = "early"
        else:
            model = self.model_late
            model_used = "late"
        
        # Get features for target date
        row = self.df[self.df['Date'] == target_dt]
        if row.empty:
            raise ValueError(f"No data found for date {target_date}")
        
        # Get feature columns (same as training)
        feature_cols = self._get_feature_columns()
        X = row[feature_cols].values
        
        # Predict
        probability = float(model.predict_proba(X)[0, 1])
        
        # Determine risk level
        risk_level = self._classify_risk(probability)
        
        # Confidence
        confidence = abs(probability - 0.5) * 2
        
        return CrisisPrediction(
            date=target_date,
            probability=probability,
            model_used=model_used,
            risk_level=risk_level,
            confidence=float(confidence)
        )
    
    def _get_feature_columns(self):
        """Get feature columns (excluding targets and non-numeric)"""
        exclude = [
            'Date', 'Year', 'News_Headline', 'News_Body', 'Event_Type',
            'Event_ID', 'Affected_Sector', 'Shock_Active', 'Regime',
            'Future_MDD', 'Target_Crisis', 'Target_Volatility', 'Log_Ret'
        ]
        feats = [c for c in self.df.columns if c not in exclude]
        return [c for c in feats if pd.api.types.is_numeric_dtype(self.df[c])]
    
    def _classify_risk(self, prob: float) -> str:
        """Classify risk level from probability"""
        if prob < 0.2:
            return "Low"
        elif prob < 0.5:
            return "Medium"
        elif prob < 0.8:
            return "High"
        else:
            return "Critical"


class VolatilityPredictionTool(BaseTool):
    """
    Predict volatility using LSTM model
    """
    
    def __init__(self):
        super().__init__()
        self.df = pd.read_parquet(DATA_PATH)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Prepare features
        self._prepare_features()
        
        # Get feature columns
        self.feature_cols = self._get_feature_columns()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Load LSTM model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(self.feature_cols)
        self.model = LSTMWithAttention(input_dim, LSTM_HIDDEN_DIM, LSTM_ATTENTION_DIM)
        self.model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    @property
    def name(self) -> str:
        return "get_volatility"
    
    @property
    def description(self) -> str:
        return """Predict future realized volatility using LSTM model.
        Returns annualized volatility and regime classification (Low/Normal/High/Extreme)."""
    
    @property
    def input_schema(self) -> type[VolatilityInput]:
        return VolatilityInput
    
    def _execute(self, target_date: date) -> VolatilityPrediction:
        """
        Predict volatility
        
        Args:
            target_date: Date to predict
            
        Returns:
            VolatilityPrediction with volatility and regime
        """
        target_dt = pd.to_datetime(target_date)
        
        # Get sequence ending at target_date
        end_idx = self.df[self.df['Date'] == target_dt].index
        if len(end_idx) == 0:
            raise ValueError(f"No data found for date {target_date}")
        
        end_idx = end_idx[0]
        start_idx = max(0, end_idx - LSTM_SEQ_LENGTH + 1)
        
        if end_idx - start_idx + 1 < LSTM_SEQ_LENGTH:
            raise ValueError(f"Insufficient historical data (need {LSTM_SEQ_LENGTH} days)")
        
        # Extract sequence
        seq_df = self.df.iloc[start_idx:end_idx+1]
        X_raw = seq_df[self.feature_cols].values
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Reshape for LSTM
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(X_tensor)
            volatility = float(pred.cpu().numpy()[0, 0])
        
        # Classify regime
        regime = self._classify_volatility_regime(volatility)
        
        return VolatilityPrediction(
            date=target_date,
            predicted_volatility=volatility,
            volatility_regime=regime
        )
    
    def _prepare_features(self):
        """Add technical indicators (same as training)"""
        price_col = 'Asset_01_TEC_Close'
        if price_col not in self.df.columns:
            price_cols = [c for c in self.df.columns if 'Close' in c]
            price_col = price_cols[0]
        
        # RSI
        delta = self.df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        self.df['Tech_RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.df[price_col].ewm(span=12, adjust=False).mean()
        exp2 = self.df[price_col].ewm(span=26, adjust=False).mean()
        self.df['Tech_MACD'] = exp1 - exp2
        
        # MA Distance
        ma_50 = self.df[price_col].rolling(window=50).mean()
        self.df['Tech_Dist_MA50'] = (self.df[price_col] - ma_50) / ma_50
        
        # Bollinger Bands
        bb_mean = self.df[price_col].rolling(20).mean()
        bb_std = self.df[price_col].rolling(20).std()
        self.df['Tech_BB_Upper'] = bb_mean + (bb_std * 2)
        self.df['Tech_BB_Lower'] = bb_mean - (bb_std * 2)
        self.df['Tech_BB_Pos'] = (self.df[price_col] - self.df['Tech_BB_Lower']) / \
                                  (self.df['Tech_BB_Upper'] - self.df['Tech_BB_Lower'] + 1e-6)
        
        # Macro features
        if 'Logistics' in self.df.columns and 'Production' in self.df.columns:
            self.df['Macro_SupplyChain_Stress'] = self.df['Logistics'] / (self.df['Production'] + 1e-6)
        
        if 'Interest_Rate' in self.df.columns and 'Target_Rate' in self.df.columns:
            self.df['Macro_Rate_Gap'] = self.df['Interest_Rate'] - self.df['Target_Rate']
            self.df['Macro_Rate_ROC'] = self.df['Interest_Rate'].diff()
        
        # Regime encoding
        if 'Regime' in self.df.columns:
            dummies = pd.get_dummies(self.df['Regime'], prefix='Regime')
            self.df = pd.concat([self.df, dummies], axis=1)
        
        self.df = self.df.dropna().reset_index(drop=True)
    
    def _get_feature_columns(self):
        """Get feature columns (same as training)"""
        exclude = [
            'Date', 'Year', 'News_Headline', 'News_Body', 'Event_Type',
            'Event_ID', 'Affected_Sector', 'Shock_Active', 'Regime',
            'Future_MDD', 'Target_Crisis', 'Target_Volatility', 'Log_Ret'
        ]
        feats = [c for c in self.df.columns if c not in exclude]
        return [c for c in feats if pd.api.types.is_numeric_dtype(self.df[c])]
    
    def _classify_volatility_regime(self, vol: float) -> str:
        """Classify volatility regime"""
        if vol < 0.15:
            return "Low"
        elif vol < 0.25:
            return "Normal"
        elif vol < 0.40:
            return "High"
        else:
            return "Extreme"