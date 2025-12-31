import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import uuid
import json
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DTYPE = np.float64
TRADING_DAYS = 252
YEARS = 10
TOTAL_STEPS = int(TRADING_DAYS * YEARS)
DT = 1 / TRADING_DAYS
MAX_RETRIES = 5

class MarketRegime(Enum):
    GROWTH = "Growth"        
    SHOCK = "Shock"      
    RECOVERY = "Recovery"     
    OVERHEATING = "Overheating" 
    INTERVENTION = "Intervention" 
    STABILIZATION = "Stabilization" 

class Sector(Enum):
    TECH = "Technology"
    INDUSTRIAL = "Industrial"
    FINANCE = "Finance"
    ENERGY = "Energy"
    HEALTH = "Healthcare"

class EventType(Enum):
    MONETARY = "Monetary Policy"    
    MACRO_GDP = "GDP Report"   
    MACRO_UNEMP = "Unemployment Data"
    GEOPOLITICAL = "Geopolitical"  
    SUPPLY_CHAIN = "Supply Chain"
    EARNINGS = "Earnings Announcement"
    CREDIT_RATING = "Credit Rating Change"

@dataclass
class SimulationConfig:
    seed: int = 42
    num_assets: int = 10 
    start_price: float = 100.0
    start_rate: float = 0.02
    start_unemployment: float = 0.04 
    start_date: str = "2024-01-01"
    output_dir: str = "data/" 
    parquet_filename: str = "stochastic_market_data.parquet"
    calc_technical_indicators: bool = False

class MathUtils:
    """Advanced Financial Mathematics and Statistics Library."""
    
    @staticmethod
    def runge_kutta_4(func, y0: np.ndarray, t: np.ndarray, args: tuple) -> np.ndarray:
        """Fourth-order Runge-Kutta solver for Lotka-Volterra system."""
        n = len(t)
        y = np.zeros((n, len(y0)))
        y[0] = y0
        h = t[1] - t[0]
        
        for i in range(n - 1):
            k1 = func(t[i], y[i], *args)
            k2 = func(t[i] + h/2, y[i] + h*k1/2, *args)
            k3 = func(t[i] + h/2, y[i] + h*k2/2, *args)
            k4 = func(t[i] + h, y[i] + h*k3, *args)
            y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        return y

    @staticmethod
    def calculate_rs_hurst(series: np.ndarray) -> float:
        """
        Hurst Exponent calculation via R/S Analysis.
        """
        series = np.array(series)
        if len(series) < 100: 
            return 0.5
        
        returns = np.diff(np.log(series + 1e-9))
        
        min_k = 4
        max_k = int(np.floor(np.log2(len(returns))))
        R_S_dict = []
        
        for k in range(min_k, max_k):
            n = 2**k
            chunks = [returns[i:i+n] for i in range(0, len(returns), n) if len(returns[i:i+n])==n]
            rs_values = []
            
            for chunk in chunks:
                mean = np.mean(chunk)
                y = chunk - mean
                z = np.cumsum(y)
                R = np.max(z) - np.min(z)
                S = np.std(chunk, ddof=1)
                
                if S < 1e-9: continue
                rs_values.append(R/S)
                
            if rs_values:
                R_S_dict.append((np.log(n), np.log(np.mean(rs_values))))
        
        if len(R_S_dict) < 3: 
            return 0.5
            
        x = [p[0] for p in R_S_dict]
        y = [p[1] for p in R_S_dict]
        
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    @staticmethod
    def arch_lm_test(returns: np.ndarray, lags: int = 10) -> Tuple[bool, float]:
        """Engle's ARCH-LM Test for volatility clustering."""
        returns = returns[~np.isnan(returns)]
        returns = returns - np.mean(returns)
        sq_returns = returns ** 2
        
        n = len(sq_returns)
        X = np.zeros((n - lags, lags))
        
        for i in range(lags):
            X[:, i] = sq_returns[i:n-lags+i]
        
        y = sq_returns[lags:]
        
        X_with_const = np.column_stack([np.ones(len(y)), X])
        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            residuals = y - X_with_const @ beta
            ssr = np.sum(residuals ** 2)
            
            tss = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ssr / tss) if tss > 0 else 0
            
            lm_stat = len(y) * r_squared
            critical_value = stats.chi2.ppf(0.95, lags)
            
            has_arch = lm_stat > critical_value
            
            return has_arch, lm_stat
        except:
            autocorrs = [pd.Series(sq_returns).autocorr(lag=i) for i in range(1, lags+1)]
            return np.mean(autocorrs) > 0.02, 0.0

class TechnicalIndicators:
    """Feature Engineering Layer - Technical Indicators Suite."""
    
    @staticmethod
    def add_indicators(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """Compute comprehensive technical indicators for ML models."""
        
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        exp1 = df[price_col].ewm(span=12, adjust=False).mean()
        exp2 = df[price_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df['BB_Mid'] = df[price_col].rolling(window=20).mean()
        df['BB_Std'] = df[price_col].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
        df['Volatility_20d'] = df[price_col].pct_change().rolling(20).std() * np.sqrt(252)
        
        if 'High' in df.columns and 'Low' in df.columns:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df[price_col].shift())
            low_close = np.abs(df['Low'] - df[price_col].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR_14'] = true_range.rolling(window=14).mean()
        
        df['Momentum_10'] = df[price_col].diff(10)
        df['ROC_10'] = df[price_col].pct_change(periods=10) * 100
        
        return df.fillna(0)

class CorpusGenerator:
    """Text generation engine for GraphRAG corpus."""
    
    @staticmethod
    def generate_fomc_minutes(rate: float, prev_rate: float, regime: str) -> Tuple[str, str, float]:
        """Generate Federal Reserve FOMC meeting minutes."""
        delta = rate - prev_rate
        action = "raise" if delta > 0.001 else "cut" if delta < -0.001 else "maintain"
        tone = "hawkish" if delta > 0 else "dovish" if delta < 0 else "neutral"
        
        impact = -1.2 if delta > 0 else 1.0 if delta < 0 else 0.0
        
        headline = f"FOMC Decision: Target Rate Set to {rate*100:.2f}%"
        body = (f"The Federal Open Market Committee (FOMC) decided to {action} the target range. "
                f"Committee members noted that economic conditions are currently reflecting {regime.lower()}. "
                f"Inflation pressure remains a key determinant in this policy path. "
                f"Market sentiment analysis suggests a {tone.upper()} stance from the chair. "
                f"Quantitative tightening measures were also discussed. "
                f"Forward guidance indicates {'continued vigilance' if delta > 0 else 'accommodative stance'}.")
        
        return headline, body, impact

    @staticmethod
    def generate_geopolitical_event(sector: Sector) -> Tuple[str, str, float]:
        """Generate geopolitical crisis event affecting specific sector."""
        headline = f"BREAKING: New Trade Sanctions Hit {sector.value} Sector"
        body = (f"Government authorities have announced strict trade restrictions targeting the {sector.value} industry. "
                f"Global supply chains are expected to be severely disrupted due to new embargoes. "
                f"Major analysts predict a 15-20% revenue drop for key players in the region. "
                f"Diplomatic tensions are escalating significantly. "
                f"S&P analysts downgraded sector outlook from stable to negative.")
        impact = -2.5
        return headline, body, impact

    @staticmethod
    def generate_logistics_crisis() -> Tuple[str, str, float]:
        """Generate global supply chain disruption event."""
        headline = "Global Supply Chain Crisis Worsens: Bottlenecks Everywhere"
        body = ("Shipping lanes are congested causing massive delays in raw material delivery. "
                "Industrial and Technology sectors are facing critical shortages of semiconductors and raw steel. "
                "Production costs are skyrocketing, putting pressure on consumer prices (CPI). "
                "Freight indices have hit all-time highs. Port congestion metrics show 3-week delays. "
                "Manufacturing PMI dropped below 50 indicating contraction.")
        impact = -1.5
        return headline, body, impact

    @staticmethod
    def generate_gdp_report(growth_rate: float, regime: str) -> Tuple[str, str, float]:
        """Generate quarterly GDP growth report."""
        sentiment = "robust" if growth_rate > 0.02 else "sluggish" if growth_rate > 0 else "contractionary"
        headline = f"Quarterly GDP Report: Economy Shows {sentiment.title()} Performance"
        body = (f"The latest Gross Domestic Product report indicates a growth rate of {growth_rate*100:.2f}%. "
                f"This figure aligns with the current {regime.lower()} phase of the market cycle. "
                f"Analysts are debating whether this trend is sustainable in the long run given the current yield curve. "
                f"Consumer spending {'accelerated' if growth_rate > 0.015 else 'decelerated'} from previous quarter. "
                f"Business investment showed {'strength' if growth_rate > 0.01 else 'weakness'}.")
        impact = 1.2 if growth_rate > 0.02 else 0.8 if growth_rate > 0 else -1.2
        return headline, body, impact

    @staticmethod
    def generate_unemployment_report(rate: float, prev_rate: float, gdp_growth: float) -> Tuple[str, str, float]:
        """Generate unemployment report with GDP correlation."""
        delta = rate - prev_rate
        trend = "rising" if delta > 0 else "falling"
        
        consistency = "consistent" if (delta > 0 and gdp_growth < 0.01) or (delta < 0 and gdp_growth > 0.01) else "diverging"
        
        headline = f"Labor Market Update: Unemployment Rate at {rate*100:.1f}%"
        body = (f"The Bureau of Labor Statistics released new data showing a {trend} unemployment rate. "
                f"Job creation numbers came in {('below' if delta > 0 else 'above')} expectations. "
                f"This signals potential shifts in consumer spending power and wage inflation. "
                f"The labor market dynamics are {consistency} with current GDP trends. "
                f"Participation rate {'decreased' if delta > 0 else 'increased'}, suggesting "
                f"{'discouraged workers' if delta > 0 else 'workforce re-entry'}.")
        
        impact = -0.8 if delta > 0 else 0.5
        if consistency == "diverging":
            impact *= 1.3
            
        return headline, body, impact

    @staticmethod
    def generate_earnings_announcement(sector: Sector, surprise_pct: float) -> Tuple[str, str, float]:
        """Generate earnings surprise announcement - NEW"""
        sentiment = "Beat" if surprise_pct > 0 else "Miss"
        direction = "above" if surprise_pct > 0 else "below"
        
        headline = f"{sector.value} Sector Earnings {sentiment}: {abs(surprise_pct):.1f}% {direction.title()} Expectations"
        body = (f"Major {sector.value} companies reported quarterly earnings {direction} analyst forecasts. "
                f"The surprise factor of {abs(surprise_pct):.1f}% has caught market participants off-guard. "
                f"Revenue guidance for next quarter {'raised' if surprise_pct > 0 else 'lowered'} by management. "
                f"Margin pressure from {'expansion' if surprise_pct > 0 else 'contraction'} in operating efficiency. "
                f"Institutional investors are {'accumulating' if surprise_pct > 0 else 'reducing'} positions.")
        
        # Scaled impact: ±10% surprise = ±0.5% immediate price impact
        impact = surprise_pct * 0.05
        return headline, body, impact

    @staticmethod
    def generate_credit_rating_change(sector: Sector, downgrade: bool = True) -> Tuple[str, str, float]:
        """Generate credit rating change event - NEW"""
        action = "Downgrade" if downgrade else "Upgrade"
        old_rating = "A" if downgrade else "BBB"
        new_rating = "BBB" if downgrade else "A"
        
        headline = f"Credit Rating {action}: {sector.value} Sector {old_rating} → {new_rating}"
        body = (f"Major rating agency has {'downgraded' if downgrade else 'upgraded'} {sector.value} sector debt. "
                f"The change from {old_rating} to {new_rating} reflects {'deteriorating' if downgrade else 'improving'} fundamentals. "
                f"Leverage ratios have {'increased' if downgrade else 'decreased'} beyond acceptable thresholds. "
                f"Interest coverage metrics show {'weakness' if downgrade else 'strength'} in debt servicing capability. "
                f"Bond yields are expected to {'widen' if downgrade else 'tighten'} significantly.")
        
        impact = -1.5 if downgrade else 1.2
        return headline, body, impact

class MacroEngine:
    """Macroeconomic simulation engine implementing stochastic differential equations."""
    
    def __init__(self, steps: int, dt: float):
        self.steps = steps
        self.dt = dt

    def run_vasicek_extended(self, r0: float, target_rates: np.ndarray, 
                            regime_vol_mult: np.ndarray, gdp_growth: np.ndarray = None) -> np.ndarray:
        """Extended Vasicek/OU Hybrid Model with regime-dependent parameters."""
        rates = np.zeros(self.steps)
        rates[0] = r0
        dw = np.random.normal(0, np.sqrt(self.dt), self.steps)
        
        base_a = 0.8
        base_sigma = 0.015
        
        for t in range(1, self.steps):
            b = target_rates[t-1]
            current_sigma = base_sigma * regime_vol_mult[t-1]
            current_a = base_a * 2.0 if regime_vol_mult[t-1] > 1.5 else base_a
            
            gdp_adjustment = 0.0
            if gdp_growth is not None:
                gdp_adjustment = 0.002 * gdp_growth[t-1] if gdp_growth[t-1] > 0.03 else 0.0
            
            dr = current_a * (b - rates[t-1] + gdp_adjustment) * self.dt + current_sigma * dw[t]
            rates[t] = max(rates[t-1] + dr, 0.0)
            
        return rates

    @staticmethod
    def _lotka_volterra_deriv(t: float, y: np.ndarray, alpha: float, beta: float, 
                              delta: float, gamma: float) -> np.ndarray:
        """
        Lotka-Volterra derivatives with seasonal effects.
        
        t parameter enables quarterly seasonality modeling:
        - Q4 (Oct-Dec): Holiday demand spike
        - Q1 (Jan-Mar): Post-holiday slowdown
        """
        L, P = y

        seasonal_factor = 1.0 + 0.10 * np.sin(2 * np.pi * t / 63)
        
        eff_alpha = alpha * seasonal_factor
        
        dL = eff_alpha * L - beta * L * P
        dP = delta * L * P - gamma * P
        return np.array([dL, dP])

    def run_lotka_volterra_rk4(self, L0: float, P0: float, 
                               shock_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run Lotka-Volterra system with RK4 integration."""
        alpha, beta = 1.0, 1.0
        delta, gamma = 1.0, 1.0
        
        L = np.zeros(self.steps)
        P = np.zeros(self.steps)
        L[0], P[0] = L0, P0
        
        for t in range(0, self.steps - 1):
            current_y = np.array([L[t], P[t]])
            
            shock_factor = 0.3 if shock_indices[t] else 1.0
            eff_alpha = alpha * shock_factor
            
            h = self.dt
            k1 = self._lotka_volterra_deriv(t, current_y, eff_alpha, beta, delta, gamma)
            k2 = self._lotka_volterra_deriv(t + h/2, current_y + h*k1/2, eff_alpha, beta, delta, gamma)
            k3 = self._lotka_volterra_deriv(t + h/2, current_y + h*k2/2, eff_alpha, beta, delta, gamma)
            k4 = self._lotka_volterra_deriv(t + h, current_y + h*k3, eff_alpha, beta, delta, gamma)
            
            next_y = current_y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            next_y += np.random.normal(0, 0.005, 2)
            
            L[t+1] = max(next_y[0], 0.1)
            P[t+1] = max(next_y[1], 0.1)

        gdp_proxy = (L * P)
        gdp_growth = pd.Series(gdp_proxy).pct_change().rolling(60).mean().fillna(0).values * 2
        
        return L, P, gdp_growth
    
    def compute_unemployment_dynamics(self, gdp_growth: np.ndarray, 
                                     interest_rates: np.ndarray,
                                     regime_vol: np.ndarray,
                                     start_unemp: float) -> np.ndarray:
        """Compute unemployment rate dynamics using Okun's Law + monetary policy feedback."""
        unemployment = np.zeros(self.steps)
        unemployment[0] = start_unemp
        
        trend_growth = 0.02
        
        for t in range(1, self.steps):
            gdp_effect = -0.5 * (gdp_growth[t-1] - trend_growth)
            
            lag = 126
            rate_effect = 0.0
            if t > lag:
                rate_change = interest_rates[t-lag] - interest_rates[max(0, t-lag-63)]
                rate_effect = 0.3 * rate_change
            
            stress_effect = 0.001 * (regime_vol[t-1] - 1.0)
            shock = np.random.normal(0, 0.0015)
            
            change = gdp_effect + rate_effect + stress_effect + shock
            unemployment[t] = np.clip(unemployment[t-1] + change, 0.03, 0.18)
        
        return unemployment

class DynamicBetaEngine:
    """
    This allows causal models to discover true relationships from data.
    """
    
    def __init__(self, sectors: List[Sector], steps: int, dt: float):
        self.sectors = sectors
        self.steps = steps
        self.dt = dt
        
        self.base_betas = {
            Sector.TECH:      {'rate': 1.8, 'gdp': 1.6, 'logistics': 0.9, 'unemp': 1.2},
            Sector.FINANCE:   {'rate': 2.2, 'gdp': 1.1, 'logistics': 0.3, 'unemp': 0.8},
            Sector.INDUSTRIAL:{'rate': 1.2, 'gdp': 1.3, 'logistics': 1.8, 'unemp': 1.4},
            Sector.ENERGY:    {'rate': 0.6, 'gdp': 1.0, 'logistics': 1.2, 'unemp': 0.5},
            Sector.HEALTH:    {'rate': 0.8, 'gdp': 0.5, 'logistics': 0.5, 'unemp': 0.3}
        }
        
        self.beta_history = {sec: {factor: np.zeros(steps) for factor in ['rate', 'gdp', 'logistics', 'unemp']} 
                             for sec in sectors}
        
    def generate_dynamic_betas(self, regime_vol: np.ndarray) -> None:
        """
        Generate time-varying betas using OU process for each factor.
        
        MODEL: dβ_t = κ(θ - β_t)dt + σ_β dW_t
        
        Where:
            κ: Mean reversion speed (sector adapts to new equilibrium)
            θ: Long-term mean (base beta)
            σ_β: Beta volatility (increases in crisis)
        """
        for sector in self.sectors:
            base = self.base_betas[sector]
            
            for factor in ['rate', 'gdp', 'logistics', 'unemp']:
                beta_series = np.zeros(self.steps)
                beta_series[0] = base[factor]
                
                kappa = 0.15 
                theta = base[factor]
                sigma_beta = 0.08 
                
                dw = np.random.normal(0, np.sqrt(self.dt), self.steps)
                
                for t in range(1, self.steps):
                    current_sigma = sigma_beta * regime_vol[t-1]
                    
                    d_beta = kappa * (theta - beta_series[t-1]) * self.dt + current_sigma * dw[t]
                    beta_series[t] = max(beta_series[t-1] + d_beta, 0.1) 
                
                self.beta_history[sector][factor] = beta_series
    
    def get_beta(self, sector: Sector, factor: str, time_step: int) -> float:
        """Retrieve beta coefficient at specific time step."""
        return self.beta_history[sector][factor][time_step]

class AssetEngine:
    """Multi-asset price generation with dynamic causality and market microstructure."""
    
    def __init__(self, steps: int, dt: float, n_assets: int, config: SimulationConfig):
        self.steps = steps
        self.dt = dt
        self.n = n_assets
        self.cfg = config
        
        sector_list = list(Sector)
        self.sectors = [sector_list[i % len(sector_list)] for i in range(n_assets)]
        self.names = [f"Asset_{i+1:02d}_{self.sectors[i].value[:3].upper()}" for i in range(n_assets)]
        
        self.beta_engine = DynamicBetaEngine(self.sectors, steps, dt)
        
        self.base_corr = self._build_base_correlation()
        
        self.tick_size = 0.01 
        self.base_spread = 0.0005

    def _build_base_correlation(self) -> np.ndarray:
        """Construct base correlation matrix with sector clustering."""
        corr = np.eye(self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.sectors[i] == self.sectors[j]:
                    rho = np.random.uniform(0.70, 0.90)
                else:
                    rho = np.random.uniform(0.1, 0.3)
                corr[i, j] = rho
                corr[j, i] = rho
        
        min_eig = np.min(np.linalg.eigvals(corr))
        if min_eig < 0:
            corr -= 1.1 * min_eig * np.eye(self.n)
        
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        return corr

    def _get_dynamic_cholesky(self, stress_factor: float) -> np.ndarray:
        """Compute dynamic Cholesky decomposition with correlation breakdown."""
        if stress_factor <= 1.0:
            return np.linalg.cholesky(self.base_corr)
        
        alpha = min(0.8, (stress_factor - 1.0) * 0.25)
        ones_matrix = np.ones((self.n, self.n))
        stressed_corr = (1 - alpha) * self.base_corr + alpha * ones_matrix
        
        np.fill_diagonal(stressed_corr, 1.0)
        
        min_eig = np.min(np.linalg.eigvals(stressed_corr))
        if min_eig < 0:
            stressed_corr -= 1.05 * min_eig * np.eye(self.n)
            
        return np.linalg.cholesky(stressed_corr)

    def generate_prices(self, 
                       interest_rates: np.ndarray, 
                       gdp_growth: np.ndarray, 
                       logistics_stress: np.ndarray,
                       unemployment: np.ndarray,
                       regime_vol: np.ndarray,
                       shock_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate asset prices with CAUSAL LAGS and Noise Injection.
        """
        self.beta_engine.generate_dynamic_betas(regime_vol)
        
        prices = np.zeros((self.steps, self.n))
        prices[0] = self.cfg.start_price
        
        uncorrelated_dW = np.random.normal(0, np.sqrt(self.dt), (self.n, self.steps))
        
        jump_mean = -0.10
        jump_std = 0.15
        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        
        LAG_RATE = 10   
        LAG_GDP = 63    
        LAG_LOGISTICS = 5
        LAG_UNEMP = 21
        
        for t in range(1, self.steps):
            current_vol_mult = regime_vol[t]
            
            idx_rate = max(0, t - LAG_RATE)
            idx_gdp = max(0, t - LAG_GDP)
            idx_log = max(0, t - LAG_LOGISTICS)
            idx_unemp = max(0, t - LAG_UNEMP)
            
            L_chol = self._get_dynamic_cholesky(stress_factor=current_vol_mult)
            correlated_noise = L_chol @ uncorrelated_dW[:, t]
            
            for i in range(self.n):
                sec = self.sectors[i]
                
                beta_rate = self.beta_engine.get_beta(sec, 'rate', t)
                beta_gdp = self.beta_engine.get_beta(sec, 'gdp', t)
                beta_logistics = self.beta_engine.get_beta(sec, 'logistics', t)
                beta_unemp = self.beta_engine.get_beta(sec, 'unemp', t)

                macro_drift = 0.08 \
                              - (beta_rate * interest_rates[idx_rate]) \
                              + (beta_gdp * gdp_growth[idx_gdp]) \
                              - (beta_unemp * (unemployment[idx_unemp] - 0.04))
                
                logistics_penalty = 0.0
                if logistics_stress[idx_log] < 0.6:
                    logistics_penalty = -0.08 * beta_logistics

                event_shock = shock_matrix[t-1, i] * 0.15 

                sigma = 0.22 * current_vol_mult

                lambda_jump = 0.5 * (current_vol_mult ** 2) if current_vol_mult > 1.2 else 0.05

                jump_val = 0.0
                if np.random.random() < lambda_jump * self.dt:
                    jump_val = np.random.normal(jump_mean, jump_std)

                current_mu = macro_drift + logistics_penalty
                compensator = lambda_jump * k
                
                drift_term = (current_mu - compensator - 0.5 * sigma**2) * self.dt
                diffusion_term = sigma * correlated_noise[i]
                shock_term = event_shock * self.dt
                
                d_ln_S = drift_term + diffusion_term + jump_val + shock_term
                
                prices[t, i] = prices[t-1, i] * np.exp(d_ln_S)

        opens, highs, lows, volumes = self._simulate_intraday_microstructure(prices, regime_vol)
        
        return {
            "Close": prices, 
            "Open": opens, 
            "High": highs, 
            "Low": lows, 
            "Volume": volumes
        }

    def _simulate_intraday_microstructure(self, closes: np.ndarray, regime_vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate realistic intraday OHLV with market microstructure.
        
        Improvements:
        - Bid-ask spread modeling
        - Tick size rounding
        - Overnight gap from news
        - Volume-price momentum relationship (herding)
        """
        n_steps, n_assets = closes.shape
        vol_factor = regime_vol[:, np.newaxis]
        
        spread = self.base_spread * vol_factor
        
        overnight_gap = np.random.normal(0, 0.003 * vol_factor, (n_steps, n_assets))
        opens = np.roll(closes, 1, axis=0) * (1 + overnight_gap)
        opens[0] = closes[0] 
        
        opens = np.round(opens / self.tick_size) * self.tick_size
        
        daily_range = 0.02 * vol_factor

        high_excursion = np.abs(np.random.gamma(2, 1, (n_steps, n_assets))) * daily_range / 2
        highs = np.maximum(opens, closes) * (1 + high_excursion + spread)
        highs = np.round(highs / self.tick_size) * self.tick_size
        
        low_excursion = np.abs(np.random.gamma(2, 1, (n_steps, n_assets))) * daily_range / 2
        lows = np.minimum(opens, closes) * (1 - low_excursion - spread)
        lows = np.round(lows / self.tick_size) * self.tick_size
        
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        price_change = np.abs(closes - np.roll(closes, 1, axis=0))
        price_change[0] = 0

        momentum = price_change / (closes + 1e-8)
        herding_factor = 1 + 2 * momentum 
        
        base_volume = 1_000_000
        volumes = base_volume * vol_factor * herding_factor * np.random.gamma(4.0, 1.0, (n_steps, n_assets))
        
        return opens, highs, lows, volumes

class Orchestrator:
    """Master orchestrator with enhanced graph export."""
    
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.asset_engine = None
        self.actual_shock_impacts = {}

    def _generate_macro_schedule(self) -> pd.DataFrame:
        """Generate 10-year macroeconomic regime schedule."""
        years_arr = np.linspace(0, YEARS, TOTAL_STEPS)
        dates = pd.date_range(self.cfg.start_date, periods=TOTAL_STEPS, freq='B')
        
        df = pd.DataFrame({
            "Date": dates,
            "Year": years_arr,
            "Regime": [MarketRegime.GROWTH.value] * TOTAL_STEPS,
            "Target_Rate": [0.03] * TOTAL_STEPS,
            "Vol_Mult": [1.0] * TOTAL_STEPS,
            "Shock_Active": [False] * TOTAL_STEPS
        })
        
        mask_shock = (df["Year"] >= 3) & (df["Year"] < 4)
        df.loc[mask_shock, "Regime"] = MarketRegime.SHOCK.value
        df.loc[mask_shock, "Target_Rate"] = 0.005
        df.loc[mask_shock, "Vol_Mult"] = 4.5
        df.loc[mask_shock, "Shock_Active"] = True
        
        mask_rec = (df["Year"] >= 4) & (df["Year"] < 7)
        df.loc[mask_rec, "Regime"] = MarketRegime.RECOVERY.value
        df.loc[mask_rec, "Target_Rate"] = 0.025
        df.loc[mask_rec, "Vol_Mult"] = 1.2
        
        mask_heat = (df["Year"] >= 7) & (df["Year"] < 7.5)
        df.loc[mask_heat, "Regime"] = MarketRegime.OVERHEATING.value
        df.loc[mask_heat, "Vol_Mult"] = 2.0
        
        mask_int = (df["Year"] >= 7.5) & (df["Year"] < 8.5)
        df.loc[mask_int, "Regime"] = MarketRegime.INTERVENTION.value
        df.loc[mask_int, "Target_Rate"] = 0.09
        df.loc[mask_int, "Vol_Mult"] = 1.8
        
        mask_stab = df["Year"] >= 8.5
        df.loc[mask_stab, "Regime"] = MarketRegime.STABILIZATION.value
        df.loc[mask_stab, "Target_Rate"] = 0.045
        df.loc[mask_stab, "Vol_Mult"] = 1.0
        
        return df

    def run_with_validation(self):
        """Execute simulation with active validation loop."""
        attempt = 0
        success = False
        
        while attempt < MAX_RETRIES and not success:
            attempt += 1
            logging.info(f"Simulation Attempt {attempt}/{MAX_RETRIES}")
            
            np.random.seed(self.cfg.seed + attempt)
            random.seed(self.cfg.seed + attempt)
            
            try:
                macro_df, market_data = self._execute_single_run()
                
                is_valid = self._validate_stylized_facts(market_data["Close"])
                
                if is_valid:
                    logging.info("Validation PASSED - Stylized facts confirmed")
                    self._export_rich_data(macro_df, market_data)
                    success = True
                else:
                    logging.warning("Validation FAILED - Retrying with new seed")
                    
            except Exception as e:
                logging.error(f"Simulation crashed: {e}")
                import traceback
                traceback.print_exc()
        
        if not success:
            raise RuntimeError("CRITICAL: Could not generate valid market data after max retries")

    def _execute_single_run(self) -> Tuple[pd.DataFrame, Dict]:
        """Execute a single simulation run."""
        
        macro_df = self._generate_macro_schedule()
        
        macro_engine = MacroEngine(TOTAL_STEPS, DT)
        
        L, P, gdp = macro_engine.run_lotka_volterra_rk4(
            L0=1.2, 
            P0=0.8, 
            shock_indices=macro_df["Shock_Active"].values
        )
        macro_df["Logistics"] = L
        macro_df["Production"] = P
        macro_df["GDP_Growth"] = gdp
        
        macro_df["Interest_Rate"] = macro_engine.run_vasicek_extended(
            r0=self.cfg.start_rate,
            target_rates=macro_df["Target_Rate"].values,
            regime_vol_mult=macro_df["Vol_Mult"].values,
            gdp_growth=gdp
        )
        
        unemployment = macro_engine.compute_unemployment_dynamics(
            gdp_growth=gdp,
            interest_rates=macro_df["Interest_Rate"].values,
            regime_vol=macro_df["Vol_Mult"].values,
            start_unemp=self.cfg.start_unemployment
        )
        macro_df["Unemployment"] = unemployment
        
        macro_df, shock_matrix = self._generate_events(macro_df)
        
        self.asset_engine = AssetEngine(TOTAL_STEPS, DT, self.cfg.num_assets, self.cfg)
        
        market_data = self.asset_engine.generate_prices(
            interest_rates=macro_df["Interest_Rate"].values,
            gdp_growth=macro_df["GDP_Growth"].values,
            logistics_stress=macro_df["Logistics"].values,
            unemployment=unemployment,
            regime_vol=macro_df["Vol_Mult"].values,
            shock_matrix=shock_matrix
        )
        
        return macro_df, market_data

    def _generate_events(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate events with types (earnings, credit rating).
        """
        headlines, bodies, types, e_ids, sectors_aff = [], [], [], [], []
        shock_matrix = np.zeros((TOTAL_STEPS, self.cfg.num_assets))
        
        temp_sectors = [list(Sector)[i % 5] for i in range(self.cfg.num_assets)]
        
        prev_row = None
        current_unemp = self.cfg.start_unemployment
        
        for t, row in df.iterrows():
            eid, head, body, e_type, aff_sec = None, None, None, None, None
            impact_val = 0.0
            
            if random.random() < 0.02:
                target_sec = random.choice(list(Sector))
                surprise_pct = np.random.normal(0, 8) 
                head, body, impact_val = CorpusGenerator.generate_earnings_announcement(target_sec, surprise_pct)
                e_type = EventType.EARNINGS.value
                eid = str(uuid.uuid4())
                aff_sec = target_sec.value
                
                for i, sec in enumerate(temp_sectors):
                    if sec == target_sec:
                        shock_matrix[t, i] = impact_val
                        self.actual_shock_impacts[f"{eid}_asset_{i}"] = impact_val
            
            elif random.random() < 0.005:
                target_sec = random.choice(list(Sector))
                downgrade = random.random() < 0.7 
                head, body, impact_val = CorpusGenerator.generate_credit_rating_change(target_sec, downgrade)
                e_type = EventType.CREDIT_RATING.value
                eid = str(uuid.uuid4())
                aff_sec = target_sec.value
                
                for i, sec in enumerate(temp_sectors):
                    if sec == target_sec:
                        shock_matrix[t, i] = impact_val
                        self.actual_shock_impacts[f"{eid}_asset_{i}"] = impact_val
            
            elif random.random() < 0.004:
                target_sec = random.choice(list(Sector))
                head, body, impact_val = CorpusGenerator.generate_geopolitical_event(target_sec)
                e_type = EventType.GEOPOLITICAL.value
                eid = str(uuid.uuid4())
                aff_sec = target_sec.value
                
                for i, sec in enumerate(temp_sectors):
                    if sec == target_sec:
                        shock_matrix[t, i] = impact_val
                        self.actual_shock_impacts[f"{eid}_asset_{i}"] = impact_val
            
            elif row["Regime"] == "Shock" and (prev_row is not None and prev_row["Regime"] != "Shock"):
                head, body, impact_val = CorpusGenerator.generate_logistics_crisis()
                e_type = EventType.SUPPLY_CHAIN.value
                eid = str(uuid.uuid4())
                aff_sec = "Global"
                
                for i, sec in enumerate(temp_sectors):
                    if sec in [Sector.INDUSTRIAL, Sector.TECH]:
                        shock_matrix[t, i] = impact_val
                        self.actual_shock_impacts[f"{eid}_asset_{i}"] = impact_val
            
            elif t % 30 == 0 and t > 0:
                prev_rate = prev_row["Interest_Rate"] if prev_row is not None else 0.02
                head, body, impact_val = CorpusGenerator.generate_fomc_minutes(
                    row["Interest_Rate"], prev_rate, row["Regime"]
                )
                e_type = EventType.MONETARY.value
                eid = str(uuid.uuid4())
                aff_sec = "Finance"
                
                for i, sec in enumerate(temp_sectors):
                    if sec == Sector.FINANCE:
                        shock_matrix[t, i] = impact_val
                        self.actual_shock_impacts[f"{eid}_asset_{i}"] = impact_val
            
            elif t > 0 and t % 63 == 0:
                head, body, impact_val = CorpusGenerator.generate_gdp_report(
                    row["GDP_Growth"], row["Regime"]
                )
                e_type = EventType.MACRO_GDP.value
                eid = str(uuid.uuid4())
                aff_sec = "Global"
                
                for i in range(self.cfg.num_assets):
                    shock_matrix[t, i] = impact_val * 0.5
                    self.actual_shock_impacts[f"{eid}_asset_{i}"] = impact_val * 0.5
            
            elif t > 0 and t % 21 == 0:
                prev_unemp = current_unemp
                current_unemp = row["Unemployment"]
                head, body, impact_val = CorpusGenerator.generate_unemployment_report(
                    current_unemp, prev_unemp, row["GDP_Growth"]
                )
                e_type = EventType.MACRO_UNEMP.value
                eid = str(uuid.uuid4())
                aff_sec = "Global"
                
                for i in range(self.cfg.num_assets):
                    shock_matrix[t, i] = impact_val * 0.3
                    self.actual_shock_impacts[f"{eid}_asset_{i}"] = impact_val * 0.3
            
            headlines.append(head)
            bodies.append(body)
            types.append(e_type)
            e_ids.append(eid)
            sectors_aff.append(aff_sec)
            prev_row = row
        
        df["News_Headline"] = headlines
        df["News_Body"] = bodies
        df["Event_Type"] = types
        df["Event_ID"] = e_ids
        df["Affected_Sector"] = sectors_aff
        
        return df, shock_matrix

    def _validate_stylized_facts(self, prices: np.ndarray) -> bool:
        """Validate that generated data exhibits required Stylized Facts."""
        returns = pd.Series(prices[:, 0]).pct_change().dropna()
        
        kurt = stats.kurtosis(returns)
        fat_tails = 3.0 < kurt < 30.0
        
        has_arch, lm_stat = MathUtils.arch_lm_test(returns.values)
        
        hurst = MathUtils.calculate_rs_hurst(prices[:, 0])
        is_realistic_trend = 0.5 < hurst < 0.85
        
        logging.info(f"Validation Metrics:")
        logging.info(f"Kurtosis: {kurt:.2f} (Required: 3.0 < k < 30.0) -> {'PASS' if fat_tails else 'FAIL'}")
        logging.info(f"ARCH-LM: {lm_stat:.2f} (Present: {has_arch})")
        logging.info(f"Hurst: {hurst:.2f} (Required: 0.5 < h < 0.85) -> {'PASS' if is_realistic_trend else 'FAIL'}")
        
        return fat_tails and has_arch and is_realistic_trend

    def _export_rich_data(self, df: pd.DataFrame, market_data: Dict):
        """Export complete dataset with technical indicators and graph entities."""
        export_df = df.copy()

        asset_dfs = []
        
        for i, name in enumerate(self.asset_engine.names):
            asset_df = pd.DataFrame({
                "Close": market_data["Close"][:, i],
                "Open": market_data["Open"][:, i],
                "High": market_data["High"][:, i],
                "Low": market_data["Low"][:, i],
                "Volume": market_data["Volume"][:, i]
            })
            
            if self.cfg.calc_technical_indicators:
                asset_df = TechnicalIndicators.add_indicators(asset_df)
            
            asset_df = asset_df.add_prefix(f"{name}_")
            asset_dfs.append(asset_df)
        
        if asset_dfs:
            all_assets = pd.concat(asset_dfs, axis=1)
            export_df = pd.concat([export_df, all_assets], axis=1)
        
        output_path = os.path.join(self.cfg.output_dir, self.cfg.parquet_filename)
        export_df.to_parquet(output_path, index=False)
        logging.info(f"Exported main dataset to {output_path}")
        
        self._export_graph_entities(df, market_data["Close"])

    def _create_asset_nodes(self) -> List[Dict]:
        """Create asset node entities."""
        nodes = []
        for i, name in enumerate(self.asset_engine.names):
            nodes.append({
                "id": str(uuid.uuid4()),
                "name": name,
                "label": "Asset",
                "sector": self.asset_engine.sectors[i].value,
                "metadata": "Financial Asset Node"
            })
        return nodes
    
    def _create_event_nodes(self, df: pd.DataFrame) -> List[Dict]:
        """Create event node entities."""
        nodes = []
        valid_events = df[df["Event_ID"].notnull()]
        
        for _, row in valid_events.iterrows():
            nodes.append({
                "id": row["Event_ID"],
                "label": "Event",
                "headline": row["News_Headline"],
                "body": row["News_Body"],
                "type": row["Event_Type"],
                "date": row["Date"].strftime('%Y-%m-%d'),
                "affected_sector": row["Affected_Sector"],
                "regime": row["Regime"]
            })
        return nodes
    
    def _create_macro_nodes(self, df: pd.DataFrame) -> List[Dict]:
        """Create macro variable node entities."""
        nodes = []
        macro_vars = ["Interest_Rate", "GDP_Growth", "Unemployment", "Logistics", "Production"]
        
        for var_name in macro_vars:
            if var_name in df.columns:
                nodes.append({
                    "id": str(uuid.uuid4()),
                    "label": "MacroVariable",
                    "name": var_name,
                    "mean_value": float(df[var_name].mean()),
                    "std_value": float(df[var_name].std()),
                    "min_value": float(df[var_name].min()),
                    "max_value": float(df[var_name].max())
                })
        return nodes
    
    def _create_event_asset_edges(self, nodes_events: List[Dict], nodes_assets: List[Dict]) -> List[Dict]:
        """
        Create Event→Asset edges with sector-aware fallback.
        
        Priority:
        1. Ground truth from actual_shock_impacts
        2. Sector-based heuristic weighting
        """
        edges = []
        asset_map = {a["name"]: a["id"] for a in nodes_assets}
        asset_sectors = {a["name"]: a["sector"] for a in nodes_assets}
        
        for event in nodes_events:
            event_id = event["id"]
            event_type = event["type"]
            target_sec = event["affected_sector"]
            
            for asset_name, asset_id in asset_map.items():
                asset_sec = asset_sectors[asset_name]
                asset_idx = list(asset_map.keys()).index(asset_name)
                
                impact_key = f"{event_id}_asset_{asset_idx}"
                
                if impact_key in self.actual_shock_impacts:
                    actual_impact = self.actual_shock_impacts[impact_key]
                    weight = min(abs(actual_impact) / 3.0, 1.0)
                    
                    if weight > 0.2:
                        edges.append({
                            "source": event_id,
                            "target": asset_id,
                            "type": "AFFECTS",
                            "weight": float(weight),
                            "actual_impact": float(actual_impact),
                            "causal_lag": 0,
                            "is_ground_truth": True
                        })
                
                else:
                    weight = self._compute_sector_weight(event_type, target_sec, asset_sec)
                    
                    if weight > 0.3:
                        edges.append({
                            "source": event_id,
                            "target": asset_id,
                            "type": "AFFECTS",
                            "weight": float(weight),
                            "causal_lag": 0 if weight > 0.8 else 3,
                            "is_ground_truth": False,
                            "reason": "sector_match" if target_sec == asset_sec else "cross_sector"
                        })
        
        return edges
    
    def _compute_sector_weight(self, event_type: str, target_sec: str, asset_sec: str) -> float:
        """
        Compute edge weight based on sector relationship.
        
        Uses target_sec and asset_sec to determine causality strength.
        """
        if target_sec == "Global":
            base_weight = 0.85
            if event_type == "GDP Report" and asset_sec in ["Industrial", "Technology"]:
                return 0.92
            elif event_type == "Unemployment Data" and asset_sec in ["Finance", "Industrial"]:
                return 0.88
            return base_weight

        if target_sec == asset_sec:
            if event_type == "Earnings Announcement":
                return 0.95  
            elif event_type == "Credit Rating Change":
                return 0.93  
            elif event_type == "Geopolitical":
                return 0.90 
            return 0.88
        
        spillover_matrix = {
            ("Finance", "Technology"): 0.65,      
            ("Finance", "Industrial"): 0.70,      
            ("Technology", "Industrial"): 0.55,   
            ("Energy", "Industrial"): 0.75,     
            ("Energy", "Technology"): 0.50,  
        }
        
        key1 = (target_sec, asset_sec)
        key2 = (asset_sec, target_sec)
        
        if key1 in spillover_matrix:
            return spillover_matrix[key1]
        elif key2 in spillover_matrix:
            return spillover_matrix[key2]
        
        if event_type == "Monetary Policy":
            if asset_sec == "Finance":
                return 0.98  
            elif asset_sec == "Technology":
                return 0.75 
            return 0.60
        
        elif event_type == "Supply Chain":
            if asset_sec in ["Industrial", "Technology"]:
                return 0.85
            return 0.45
        
        return 0.25
    
    def _create_asset_correlation_edges(self, prices: np.ndarray, nodes_assets: List[Dict]) -> List[Dict]:
        """Create Asset-Asset correlation edges."""
        edges = []
        
        returns = pd.DataFrame(prices).pct_change().dropna()
        corr_matrix = returns.corr()
        
        asset_ids = [a["id"] for a in nodes_assets]
        
        for i in range(len(asset_ids)):
            for j in range(i + 1, len(asset_ids)):
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) > 0.5:
                    edges.append({
                        "source": asset_ids[i],
                        "target": asset_ids[j],
                        "type": "CO_MOVES_WITH",
                        "correlation": float(corr_val),
                        "strength": "strong" if abs(corr_val) > 0.7 else "moderate",
                        "direction": "positive" if corr_val > 0 else "negative"
                    })
        
        return edges
    
    def _create_asset_macro_edges(self, nodes_assets: List[Dict], nodes_macro: List[Dict]) -> List[Dict]:
        """Create Asset-Macro sensitivity edges using dynamic betas."""
        edges = []
        
        macro_map = {m["name"]: m["id"] for m in nodes_macro}
        asset_ids = [a["id"] for a in nodes_assets]
        
        for i, asset in enumerate(nodes_assets):
            asset_id = asset_ids[i]
            sector = self.asset_engine.sectors[i]
            
            if "Interest_Rate" in macro_map:
                beta_rate_avg = np.mean(self.asset_engine.beta_engine.beta_history[sector]['rate'])
                if beta_rate_avg > 0.5:
                    edges.append({
                        "source": asset_id,
                        "target": macro_map["Interest_Rate"],
                        "type": "SENSITIVE_TO",
                        "beta_coefficient": float(beta_rate_avg),
                        "factor": "interest_rate",
                        "direction": "negative"
                    })

            if "GDP_Growth" in macro_map:
                beta_gdp_avg = np.mean(self.asset_engine.beta_engine.beta_history[sector]['gdp'])
                if beta_gdp_avg > 0.5:
                    edges.append({
                        "source": asset_id,
                        "target": macro_map["GDP_Growth"],
                        "type": "SENSITIVE_TO",
                        "beta_coefficient": float(beta_gdp_avg),
                        "factor": "gdp_growth",
                        "direction": "positive"
                    })
            
            if "Unemployment" in macro_map:
                beta_unemp_avg = np.mean(self.asset_engine.beta_engine.beta_history[sector]['unemp'])
                if beta_unemp_avg > 0.3:
                    edges.append({
                        "source": asset_id,
                        "target": macro_map["Unemployment"],
                        "type": "SENSITIVE_TO",
                        "beta_coefficient": float(beta_unemp_avg),
                        "factor": "unemployment",
                        "direction": "negative"
                    })
        
        return edges
    
    def _export_graph_files(self, nodes_assets: List[Dict], nodes_events: List[Dict], 
                           nodes_macro: List[Dict], edges_event_asset: List[Dict],
                           edges_asset_correlation: List[Dict], edges_asset_macro: List[Dict]):
        """Export all graph entities to JSONL files."""
        
        with open(f"{self.cfg.output_dir}/nodes_assets.jsonl", "w") as f:
            for n in nodes_assets:
                f.write(json.dumps(n) + "\n")
        
        with open(f"{self.cfg.output_dir}/nodes_events.jsonl", "w") as f:
            for n in nodes_events:
                f.write(json.dumps(n) + "\n")
        
        with open(f"{self.cfg.output_dir}/nodes_macro.jsonl", "w") as f:
            for n in nodes_macro:
                f.write(json.dumps(n) + "\n")
        
        with open(f"{self.cfg.output_dir}/edges_event_asset.jsonl", "w") as f:
            for e in edges_event_asset:
                f.write(json.dumps(e) + "\n")
        
        with open(f"{self.cfg.output_dir}/edges_asset_correlation.jsonl", "w") as f:
            for e in edges_asset_correlation:
                f.write(json.dumps(e) + "\n")
        
        with open(f"{self.cfg.output_dir}/edges_asset_macro.jsonl", "w") as f:
            for e in edges_asset_macro:
                f.write(json.dumps(e) + "\n")
        
        logging.info(f"Graph entities exported:")
        logging.info(f"  {len(nodes_assets)} asset nodes")
        logging.info(f"  {len(nodes_events)} event nodes")
        logging.info(f"  {len(nodes_macro)} macro variable nodes")
        logging.info(f"  {len(edges_event_asset)} event→asset edges")
        logging.info(f"  {len(edges_asset_correlation)} asset↔asset correlation edges")
        logging.info(f"  {len(edges_asset_macro)} asset→macro sensitivity edges")
    
    def _export_graph_entities(self, df: pd.DataFrame, prices: np.ndarray):
        """
        Master orchestrator for graph entity export.
        
        MODULAR DESIGN:
        - Separate node creation methods
        - Separate edge creation methods
        - Centralized file export
        """
        nodes_assets = self._create_asset_nodes()
        nodes_events = self._create_event_nodes(df)
        nodes_macro = self._create_macro_nodes(df)
        
        edges_event_asset = self._create_event_asset_edges(nodes_events, nodes_assets)
        edges_asset_correlation = self._create_asset_correlation_edges(prices, nodes_assets)
        edges_asset_macro = self._create_asset_macro_edges(nodes_assets, nodes_macro)
        
        self._export_graph_files(
            nodes_assets, nodes_events, nodes_macro,
            edges_event_asset, edges_asset_correlation, edges_asset_macro
        )

if __name__ == "__main__":
    config = SimulationConfig()
    orchestrator = Orchestrator(config)
    orchestrator.run_with_validation()