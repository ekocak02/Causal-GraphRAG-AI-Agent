import os
import shutil
import logging
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import (
    load_and_preprocess_data, 
    feature_engineering_common, 
    create_directories, 
    get_feature_columns, 
    MODELS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training Configuration
WINDOW_YEARS_TRAIN = 5
VAL_MONTHS = 6
TEST_MONTHS = 6
ROLLING_STEP_MONTHS = 3
TARGET_COL = 'Target_Volatility'

# Model Architecture
SEQ_LENGTH = 252
FORECAST_HORIZON = 10
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE_INIT = 0.001
LEARNING_RATE_FINE = 0.0001
HIDDEN_DIM = 64
ATTENTION_DIM = 32
MAX_GRAD_NORM = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VolatilityDataset(Dataset):
    """PyTorch Dataset for volatility prediction."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def create_sequences(data: np.ndarray, target: np.ndarray, seq_length: int) -> tuple:
    """Create sliding window sequences for LSTM training."""
    if len(data) <= seq_length:
        return np.array([]), np.array([])
    
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(target[i + seq_length])
    
    return np.array(xs), np.array(ys)


class LSTMWithAttention(nn.Module):
    """LSTM with attention mechanism for sequence prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_lstm, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(h_lstm), dim=1)
        context = torch.sum(attn_weights * h_lstm, dim=1)
        return self.fc(context)


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> float:
    """Execute one training epoch with gradient clipping."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss


def _evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Evaluate model on validation/test data."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            total_loss += criterion(model(X_batch), y_batch).item()
    
    return total_loss / len(loader) if len(loader) > 0 else float('inf')


def _prepare_window_data(df, start, end, feature_cols, scaler=None, fit_scaler=False, buffer_size=0):
    """Prepare scaled data for a time window with optional buffer."""
    mask = (df['Date'] >= start) & (df['Date'] < end)
    
    if mask.sum() == 0:
        return None, None, None, scaler
    
    if buffer_size > 0:
        start_idx = df[mask].index[0]
        end_idx = df[mask].index[-1] + 1
        buffer_start = max(0, start_idx - buffer_size)
        window_df = df.iloc[buffer_start:end_idx]
    else:
        window_df = df[mask]
    
    X_raw = window_df[feature_cols].values
    y_raw = window_df[TARGET_COL].values
    
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw) if scaler else X_raw
    
    return X_scaled, y_raw, window_df, scaler


def _train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    model_path: str,
    patience: int = 5
) -> float:
    """Train model with early stopping and save best checkpoint."""
    best_val_loss = float('inf')
    trigger_times = 0
    
    for epoch in range(EPOCHS):
        _train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = _evaluate_model(model, val_loader, criterion)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break
    
    return best_val_loss


def train_lstm_volatility() -> None:
    """Train LSTM volatility model using rolling window strategy."""
    create_directories()
    
    logger.info("Preparing data...")
    raw_df = load_and_preprocess_data()
    df = feature_engineering_common(raw_df)
    
    feature_cols = get_feature_columns(df)
    logger.info("Features: %d, Samples: %d", len(feature_cols), len(df))
    
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    current_start = start_date
    
    iteration = 1
    prev_model_path = None
    lstm_dir = os.path.join(MODELS_DIR, 'LSTM')
    
    logger.info("LSTM Rolling Window Training Started")
    
    while True:
        train_end = current_start + relativedelta(years=WINDOW_YEARS_TRAIN)
        val_end = train_end + relativedelta(months=VAL_MONTHS)
        test_end = val_end + relativedelta(months=TEST_MONTHS)
        
        if test_end > end_date:
            logger.info("Reached end of dataset.")
            break

        # Prepare training data
        X_train_scaled, y_train_raw, train_df, scaler = _prepare_window_data(
            df, current_start, train_end, feature_cols, fit_scaler=True
        )
        
        if X_train_scaled is None or len(train_df) <= SEQ_LENGTH:
            logger.warning("Iter %d: Insufficient training data.", iteration)
            break

        # Prepare validation data with buffer
        X_val_scaled, y_val_raw, val_df, _ = _prepare_window_data(
            df, train_end, val_end, feature_cols, scaler=scaler, buffer_size=SEQ_LENGTH
        )
        
        if X_val_scaled is None:
            break

        # Prepare test data with buffer
        X_test_scaled, y_test_raw, test_df, _ = _prepare_window_data(
            df, val_end, test_end, feature_cols, scaler=scaler, buffer_size=SEQ_LENGTH
        )

        # Create sequences
        X_train, y_train = create_sequences(X_train_scaled, y_train_raw, SEQ_LENGTH)
        X_val, y_val = create_sequences(X_val_scaled, y_val_raw, SEQ_LENGTH)
        
        if len(X_train) == 0 or len(X_val) == 0:
            current_start += relativedelta(months=ROLLING_STEP_MONTHS)
            iteration += 1
            continue

        X_test, y_test = (np.array([]), np.array([]))
        if X_test_scaled is not None and len(X_test_scaled) > SEQ_LENGTH:
            X_test, y_test = create_sequences(X_test_scaled, y_test_raw, SEQ_LENGTH)

        # Create data loaders
        train_loader = DataLoader(VolatilityDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(VolatilityDataset(X_val, y_val), batch_size=BATCH_SIZE)
        test_loader = DataLoader(VolatilityDataset(X_test, y_test), batch_size=BATCH_SIZE)
        
        # Initialize model
        input_dim = X_train.shape[2]
        model = LSTMWithAttention(input_dim, HIDDEN_DIM, ATTENTION_DIM).to(device)
        criterion = nn.MSELoss()
        
        # Fine-tuning or cold start
        if prev_model_path and os.path.exists(prev_model_path):
            model.load_state_dict(torch.load(prev_model_path))
            current_lr = LEARNING_RATE_FINE
        else:
            current_lr = LEARNING_RATE_INIT
        
        optimizer = optim.Adam(model.parameters(), lr=current_lr)
        
        logger.info("Iter %d | Train: %s-%s | LR: %s", iteration, current_start.date(), train_end.date(), current_lr)
        
        # Train with early stopping
        current_model_path = os.path.join(lstm_dir, f'lstm_vol_v{iteration}.pth')
        _train_with_early_stopping(
            model, train_loader, val_loader, optimizer, criterion, current_model_path
        )
        
        # Evaluate on test set
        if os.path.exists(current_model_path):
            model.load_state_dict(torch.load(current_model_path))
            
            if len(test_loader) > 0:
                test_loss = _evaluate_model(model, test_loader, criterion)
                rmse = np.sqrt(test_loss)
                target_mean = y_test.mean() if len(y_test) > 0 else 0
                logger.info("RESULT Test RMSE: %.4f | Target Mean: %.4f", rmse, target_mean)
            
            prev_model_path = current_model_path
        
        current_start += relativedelta(months=ROLLING_STEP_MONTHS)
        iteration += 1

    # Save final model
    if prev_model_path:
        final_path = os.path.join(lstm_dir, "lstm_vol_final.pth")
        shutil.copy(prev_model_path, final_path)
        logger.info("Final model saved: %s", final_path)

    logger.info("LSTM Volatility Training Completed.")


if __name__ == "__main__":
    train_lstm_volatility()