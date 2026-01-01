import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from datetime import date
from pathlib import Path

from agents.tools.base_tool import BaseTool
from agents.models.schemas import (
    StatisticalSummaryInput,
    CorrelationMapInput,
    DataValidationInput
)
from agents.config import DATA_PATH, ENABLE_PLOTTING, FIGURE_DPI, FIGURE_SIZE


class StatisticalSummaryTool(BaseTool):
    """
    Compute statistical summary and generate visualizations
    """
    
    def __init__(self):
        super().__init__()
        self.df = pd.read_parquet(DATA_PATH)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.plots_dir = Path("outputs/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "get_statistical_summary"
    
    @property
    def description(self) -> str:
        return """Get statistical summary (mean, std, min, max, percentiles) for any column.
        Can generate visualizations: line (time series), scatter, histogram, boxplot.
        Use for understanding distributions, trends, and outliers."""
    
    @property
    def input_schema(self) -> type[StatisticalSummaryInput]:
        return StatisticalSummaryInput
    
    def _safe_date_range(self, df: pd.DataFrame) -> Dict[str, str]:
        """NaT-safe date range calculation"""
        date_start = df['Date'].dropna().min()
        date_end = df['Date'].dropna().max()
        
        if pd.isnull(date_start) or pd.isnull(date_end):
            return {"start": "N/A", "end": "N/A"}
        return {
            "start": date_start.strftime('%Y-%m-%d'),
            "end": date_end.strftime('%Y-%m-%d')
        }
    
    def _execute(
        self,
        target_column: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        plot_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute statistical summary and optionally create plot
        
        Args:
            target_column: Column to analyze
            start_date: Filter start date
            end_date: Filter end date
            x_column: X-axis for visualization
            y_column: Y-axis for visualization (defaults to target_column)
            plot_type: Plot type (line, scatter, hist, box)
            
        Returns:
            Dict with statistics and optional plot path
        """
        # Filter data
        df = self.df.copy()
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in data")
        
        # Compute statistics
        series = df[target_column].dropna()
        
        # Calculate kurtosis
        excess_kurtosis = float(series.kurtosis())
        raw_kurtosis = excess_kurtosis + 3  
        
        # Detect potential outliers
        mean_val = float(series.mean())
        std_val = float(series.std())
        outliers_high = series[series > mean_val + 3 * std_val]
        outliers_low = series[series < mean_val - 3 * std_val]
        
        stats = {
            "column": target_column,
            "count": int(series.count()),
            "mean": mean_val,
            "std": std_val,
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "excess_kurtosis": excess_kurtosis,  
            "kurtosis": raw_kurtosis, 
            "kurtosis_interpretation": "Fat tails (leptokurtic)" if raw_kurtosis > 3 else "Normal or thin tails",
            "outliers": {
                "high_count": len(outliers_high),
                "low_count": len(outliers_low),
                "high_extreme": float(outliers_high.max()) if len(outliers_high) > 0 else None,
                "low_extreme": float(outliers_low.min()) if len(outliers_low) > 0 else None
            },
            "date_range": self._safe_date_range(df)
        }

        
        # Generate plot if requested
        plot_path = None
        if ENABLE_PLOTTING and plot_type:
            plot_path = self._create_plot(
                df, target_column, x_column, y_column or target_column, plot_type
            )
            stats["plot_path"] = str(plot_path)
        
        return stats
    
    def _create_plot(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        x_col: Optional[str],
        y_col: str,
        plot_type: str
    ) -> Path:
        """Create visualization"""
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        
        if plot_type == "line":
            x_data = df[x_col] if x_col else df.index
            ax.plot(x_data, df[y_col], linewidth=1.5)
            ax.set_xlabel(x_col or "Index")
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} Over Time")
            ax.grid(True, alpha=0.3)
            
        elif plot_type == "scatter":
            if not x_col:
                raise ValueError("scatter plot requires x_column")
            ax.scatter(df[x_col], df[y_col], alpha=0.5)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} vs {x_col}")
            ax.grid(True, alpha=0.3)
            
        elif plot_type == "hist":
            ax.hist(df[y_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(y_col)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {y_col}")
            ax.grid(True, alpha=0.3, axis='y')
            
        elif plot_type == "box":
            ax.boxplot(df[y_col].dropna())
            ax.set_ylabel(y_col)
            ax.set_title(f"Boxplot of {y_col}")
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{plot_type}_{target_col}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = self.plots_dir / filename
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path


class CorrelationMapTool(BaseTool):
    """
    Compute correlation matrix and generate heatmap
    """
    
    def __init__(self):
        super().__init__()
        self.df = pd.read_parquet(DATA_PATH)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.plots_dir = Path("outputs/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "get_corr_map"
    
    @property
    def description(self) -> str:
        return """Compute correlation matrix and generate heatmap visualization.
        Use to understand relationships between multiple variables (e.g., asset correlations)."""
    
    @property
    def input_schema(self) -> type[CorrelationMapInput]:
        return CorrelationMapInput
    
    def _execute(
        self,
        columns: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        Compute correlation matrix
        
        Args:
            columns: List of columns to correlate
            start_date: Filter start date
            end_date: Filter end date
            method: Correlation method (pearson, spearman)
            
        Returns:
            Dict with correlation matrix and plot path
        """
        # Filter data
        df = self.df.copy()
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        # Check columns exist
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        
        # Check for empty dataframe after filtering
        if len(df) == 0:
            raise ValueError("No data found for specified date range")
        
        # Compute correlation
        corr_matrix = df[columns].corr(method=method)
        
        # Generate heatmap
        plot_path = None
        if ENABLE_PLOTTING:
            plot_path = self._create_heatmap(corr_matrix, method)
        
        # NaT-safe date range calculation
        date_start = df['Date'].dropna().min()
        date_end = df['Date'].dropna().max()
        
        if pd.isnull(date_start) or pd.isnull(date_end):
            date_range_info = {"start": "N/A", "end": "N/A"}
        else:
            date_range_info = {
                "start": date_start.strftime('%Y-%m-%d'),
                "end": date_end.strftime('%Y-%m-%d')
            }
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "method": method,
            "columns": columns,
            "date_range": date_range_info,
            "plot_path": str(plot_path) if plot_path else None
        }
    
    def _create_heatmap(self, corr_matrix: pd.DataFrame, method: str) -> Path:
        """Create correlation heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10), dpi=FIGURE_DPI)
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(f"Correlation Matrix ({method.capitalize()})", fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save plot
        filename = f"corr_heatmap_{method}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = self.plots_dir / filename
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path


class DataValidationTool(BaseTool):
    """
    Validate simulation data using stylized facts (Hurst, Kurtosis, Volatility Clustering)
    """
    
    def __init__(self):
        super().__init__()
        self.df = pd.read_parquet(DATA_PATH)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
    
    @property
    def name(self) -> str:
        return "get_data_val"
    
    @property
    def description(self) -> str:
        return """Validate data quality using financial stylized facts:
        - Hurst Exponent (trend persistence)
        - Kurtosis (fat tails)
        - Volatility Clustering (GARCH effect)
        Use to assess simulation realism."""
    
    @property
    def input_schema(self) -> type[DataValidationInput]:
        return DataValidationInput
    
    def _execute(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Run validation tests
        
        Args:
            start_date: Validation period start
            end_date: Validation period end
            
        Returns:
            Dict with validation results
        """
        # Filter data
        df = self.df[
            (self.df['Date'] >= pd.to_datetime(start_date)) &
            (self.df['Date'] <= pd.to_datetime(end_date))
        ].copy()
        
        # Get first asset's log returns
        price_cols = [c for c in df.columns if 'Close' in c]
        if not price_cols:
            raise ValueError("No price columns found")
        
        price_col = price_cols[0]
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        df = df.dropna(subset=['log_return'])
        
        returns = df['log_return'].values
        
        # 1. Hurst Exponent
        hurst = self._compute_hurst(returns)
        
        # 2. Kurtosis
        kurtosis = float(pd.Series(returns).kurtosis())
        
        # 3. Volatility Clustering (Ljung-Box test on squared returns)
        squared_returns = returns ** 2
        vol_clustering = self._test_volatility_clustering(squared_returns)
        
        return {
            "period": {
                "start": start_date.strftime('%Y-%m-%d'),
                "end": end_date.strftime('%Y-%m-%d'),
                "days": len(df)
            },
            "hurst_exponent": {
                "value": float(hurst),
                "interpretation": self._interpret_hurst(hurst)
            },
            "kurtosis": {
                "value": float(kurtosis),
                "interpretation": self._interpret_kurtosis(kurtosis)
            },
            "volatility_clustering": {
                "detected": vol_clustering,
                "interpretation": "Strong GARCH effect" if vol_clustering else "Weak clustering"
            },
            "validation_passed": self._overall_validation(hurst, kurtosis, vol_clustering)
        }
    
    def _compute_hurst(self, returns: np.ndarray, max_lag: int = 20) -> float:
        """Compute Hurst Exponent using R/S method"""
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Partition series
            n_parts = len(returns) // lag
            sub_series = [returns[i*lag:(i+1)*lag] for i in range(n_parts)]
            
            rs_values = []
            for sub in sub_series:
                if len(sub) < 2:
                    continue
                mean = np.mean(sub)
                std = np.std(sub, ddof=1)
                if std == 0:
                    continue
                
                # Cumulative deviate
                cum_dev = np.cumsum(sub - mean)
                r = np.max(cum_dev) - np.min(cum_dev)
                s = std
                
                if s > 0:
                    rs_values.append(r / s)
            
            if rs_values:
                tau.append(np.mean(rs_values))
        
        # Linear regression log(tau) vs log(lag)
        log_lags = np.log(list(lags[:len(tau)]))
        log_tau = np.log(tau)
        hurst = np.polyfit(log_lags, log_tau, 1)[0]
        
        return hurst
    
    def _test_volatility_clustering(self, squared_returns: np.ndarray) -> bool:
        """Simple autocorrelation test for volatility clustering"""
        from scipy.stats import pearsonr
        
        # Check autocorrelation at lag 1
        if len(squared_returns) < 2:
            return False
        
        corr, p_value = pearsonr(squared_returns[:-1], squared_returns[1:])
        return (corr > 0.1) and (p_value < 0.05)
    
    def _interpret_hurst(self, h: float) -> str:
        if h > 0.55:
            return "Trending (persistent)"
        elif h < 0.45:
            return "Mean-reverting (anti-persistent)"
        else:
            return "Random walk (efficient market)"
    
    def _interpret_kurtosis(self, k: float) -> str:
        if k > 3:
            return f"Fat tails detected (excess kurtosis: {k-3:.2f})"
        else:
            return "Normal tails"
    
    def _overall_validation(self, hurst: float, kurtosis: float, vol_clust: bool) -> bool:
        """Check if data passes realistic financial data tests"""
        # Financial data should have: H > 0.5, K > 3, volatility clustering
        return (hurst > 0.5) and (kurtosis > 3) and vol_clust