"""
Performance Analytics Module - PRODUCTION GRADE
Tracks and analyzes trading performance with industry-standard metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json


class PerformanceAnalytics:
    """Comprehensive performance tracking and analysis"""
    
    def __init__(self, initial_capital: float):
        """
        Initialize performance analytics
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Trade history
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # Running statistics
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_duration_days = 0
        
        # Monthly tracking
        self.monthly_returns: Dict[str, float] = {}
        self.monthly_trades: Dict[str, int] = {}
        
        # Winning/losing streaks
        self.current_streak = 0
        self.max_winning_streak = 0
        self.max_losing_streak = 0
        
        # Time tracking
        self.start_time = datetime.utcnow()
        self.last_update_time = datetime.utcnow()
    
    def record_trade(self, trade: Dict):
        """
        Record a completed trade
        
        Args:
            trade: Dictionary with trade details (entry_price, exit_price, quantity, pnl, etc.)
        """
        # Add timestamp if not present
        if 'timestamp' not in trade:
            trade['timestamp'] = datetime.utcnow()
        
        # Calculate additional metrics
        if 'pnl_percent' not in trade and 'pnl' in trade and 'position_value' in trade:
            trade['pnl_percent'] = (trade['pnl'] / trade['position_value']) * 100
        
        # Store trade
        self.trades.append(trade)
        
        # Update capital
        if 'pnl' in trade:
            self.current_capital += trade['pnl']
        
        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Update streaks
        is_winner = trade.get('pnl', 0) > 0
        if is_winner:
            if self.current_streak > 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_winning_streak = max(self.max_winning_streak, self.current_streak)
        else:
            if self.current_streak < 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_losing_streak = max(self.max_losing_streak, abs(self.current_streak))
        
        # Update monthly tracking
        if 'timestamp' in trade:
            month_key = trade['timestamp'].strftime('%Y-%m')
            if month_key not in self.monthly_returns:
                self.monthly_returns[month_key] = 0.0
                self.monthly_trades[month_key] = 0
            self.monthly_returns[month_key] += trade.get('pnl', 0)
            self.monthly_trades[month_key] += 1
        
        # Record equity point
        self.equity_curve.append({
            'timestamp': trade.get('timestamp', datetime.utcnow()),
            'equity': self.current_capital,
            'drawdown': self.current_drawdown
        })
        
        self.last_update_time = datetime.utcnow()
    
    def update_equity_point(self, timestamp: Optional[datetime] = None):
        """Record current equity point (for live tracking)"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.current_capital,
            'drawdown': self.current_drawdown
        })
    
    def get_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        return (winning_trades / len(self.trades)) * 100
    
    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            float: Sharpe ratio
        """
        if len(self.trades) < 2:
            return 0.0
        
        # Calculate daily returns from trades
        returns = [t.get('pnl_percent', 0) / 100 for t in self.trades]
        
        if not returns or np.std(returns) == 0:
            return 0.0
        
        # Annualize (assuming ~252 trading days)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe = (Mean Return - Risk Free Rate) / Std Dev
        # Annualize the calculation
        sharpe = (mean_return - (risk_free_rate / 252)) / std_return * np.sqrt(252)
        
        return sharpe
    
    def get_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only penalizes downside volatility)
        
        Args:
            risk_free_rate: Annual risk-free rate
        
        Returns:
            float: Sortino ratio
        """
        if len(self.trades) < 2:
            return 0.0
        
        returns = np.array([t.get('pnl_percent', 0) / 100 for t in self.trades])
        
        # Only calculate std dev of negative returns
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        sortino = (mean_return - (risk_free_rate / 252)) / downside_std * np.sqrt(252)
        
        return sortino
    
    def get_max_drawdown_info(self) -> Dict:
        """
        Get detailed maximum drawdown information
        
        Returns:
            dict: Max drawdown details including duration and recovery
        """
        if not self.equity_curve:
            return {
                'max_drawdown_percent': 0.0,
                'max_drawdown_value': 0.0,
                'drawdown_duration_days': 0,
                'recovery_time_days': 0,
                'currently_in_drawdown': False
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.equity_curve)
        
        # Calculate running maximum
        df['running_max'] = df['equity'].cummax()
        
        # Calculate drawdown at each point
        df['drawdown_value'] = df['running_max'] - df['equity']
        df['drawdown_percent'] = (df['drawdown_value'] / df['running_max']) * 100
        
        # Find maximum drawdown
        max_dd_idx = df['drawdown_percent'].idxmax()
        max_dd_percent = df.loc[max_dd_idx, 'drawdown_percent']
        max_dd_value = df.loc[max_dd_idx, 'drawdown_value']
        
        # Find when drawdown started (last peak before max DD)
        peak_before_dd = df.loc[:max_dd_idx][df.loc[:max_dd_idx, 'drawdown_percent'] == 0]
        
        drawdown_duration_days = 0
        recovery_time_days = 0
        
        if not peak_before_dd.empty:
            dd_start_idx = peak_before_dd.index[-1]
            dd_start_time = df.loc[dd_start_idx, 'timestamp']
            dd_max_time = df.loc[max_dd_idx, 'timestamp']
            
            drawdown_duration_days = (dd_max_time - dd_start_time).days
            
            # Find recovery time (when equity reaches peak again)
            recovery_df = df.loc[max_dd_idx:]
            peak_value = df.loc[dd_start_idx, 'equity']
            recovered = recovery_df[recovery_df['equity'] >= peak_value]
            
            if not recovered.empty:
                recovery_time = recovered.iloc[0]['timestamp']
                recovery_time_days = (recovery_time - dd_max_time).days
        
        return {
            'max_drawdown_percent': float(max_dd_percent),
            'max_drawdown_value': float(max_dd_value),
            'drawdown_duration_days': drawdown_duration_days,
            'recovery_time_days': recovery_time_days,
            'currently_in_drawdown': self.current_drawdown > 0.1,
            'current_drawdown_percent': self.current_drawdown
        }
    
    def get_monthly_returns_summary(self) -> pd.DataFrame:
        """
        Get monthly returns as DataFrame
        
        Returns:
            DataFrame: Monthly returns with statistics
        """
        if not self.monthly_returns:
            return pd.DataFrame()
        
        monthly_data = []
        for month, pnl in sorted(self.monthly_returns.items()):
            # Calculate month return percentage
            # This is simplified - would need beginning of month capital for precise calculation
            return_pct = (pnl / self.initial_capital) * 100
            
            monthly_data.append({
                'month': month,
                'pnl': pnl,
                'return_pct': return_pct,
                'trades': self.monthly_trades.get(month, 0)
            })
        
        return pd.DataFrame(monthly_data)
    
    def get_trade_distribution(self) -> Dict:
        """Analyze distribution of trade returns"""
        if not self.trades:
            return {}
        
        returns = [t.get('pnl_percent', 0) for t in self.trades]
        
        return {
            'mean': np.mean(returns),
            'median': np.median(returns),
            'std': np.std(returns),
            'skewness': float(pd.Series(returns).skew()),
            'kurtosis': float(pd.Series(returns).kurtosis()),
            'percentile_25': np.percentile(returns, 25),
            'percentile_75': np.percentile(returns, 75),
            'min': np.min(returns),
            'max': np.max(returns)
        }
    
    def get_comprehensive_stats(self) -> Dict:
        """Get all performance statistics in one dictionary"""
        elapsed_time = (self.last_update_time - self.start_time).days
        if elapsed_time == 0:
            elapsed_time = 1  # Prevent division by zero

        if not self.trades:
            # Every key used by generate_performance_report() must be present
            # here too, or a zero-trade backtest/live-session crashes instead
            # of printing "no trades yet" — which is a completely normal,
            # expected outcome given how selective this strategy is.
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'total_pnl': 0.0, 'roi': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'avg_trade': 0.0,
                'profit_factor': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0,
                'max_drawdown_percent': 0.0, 'current_drawdown_percent': 0.0,
                'initial_capital': self.initial_capital, 'current_capital': self.current_capital,
                'peak_capital': self.peak_capital, 'max_winning_streak': 0,
                'max_losing_streak': 0, 'current_streak': 0,
                'avg_holding_period_hours': 0.0, 'total_days': elapsed_time,
                'trades_per_day': 0.0, 'best_trade': 0.0, 'worst_trade': 0.0,
            }
        
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Calculate holding periods
        holding_periods = []
        for t in self.trades:
            if 'entry_time' in t and 'exit_time' in t:
                duration = (t['exit_time'] - t['entry_time']).total_seconds() / 3600  # hours
                holding_periods.append(duration)
        
        # Time-based metrics
        elapsed_time = (self.last_update_time - self.start_time).days
        if elapsed_time == 0:
            elapsed_time = 1  # Prevent division by zero
        
        return {
            # Basic stats
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': self.get_win_rate(),
            
            # P&L metrics
            'total_pnl': total_pnl,
            'roi': (total_pnl / self.initial_capital) * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': total_pnl / len(self.trades) if self.trades else 0,
            'profit_factor': self.get_profit_factor(),
            
            # Risk metrics
            'sharpe_ratio': self.get_sharpe_ratio(),
            'sortino_ratio': self.get_sortino_ratio(),
            'max_drawdown_percent': self.max_drawdown,
            'current_drawdown_percent': self.current_drawdown,
            
            # Capital metrics
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            
            # Streaks
            'max_winning_streak': self.max_winning_streak,
            'max_losing_streak': self.max_losing_streak,
            'current_streak': self.current_streak,
            
            # Time metrics
            'avg_holding_period_hours': np.mean(holding_periods) if holding_periods else 0,
            'total_days': elapsed_time,
            'trades_per_day': len(self.trades) / elapsed_time if elapsed_time > 0 else 0,
            
            # Distribution
            'best_trade': max([t.get('pnl', 0) for t in self.trades]) if self.trades else 0,
            'worst_trade': min([t.get('pnl', 0) for t in self.trades]) if self.trades else 0,
        }
    
    def generate_performance_report(self, include_trades: bool = False) -> str:
        """
        Generate formatted performance report
        
        Args:
            include_trades: Include individual trade details
        
        Returns:
            str: Formatted report
        """
        stats = self.get_comprehensive_stats()
        dd_info = self.get_max_drawdown_info()
        
        report = f"""
{'='*70}
PERFORMANCE ANALYTICS REPORT
{'='*70}
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

OVERVIEW
--------
Total Trades:          {stats['total_trades']}
Trading Period:        {stats['total_days']} days
Trades per Day:        {stats['trades_per_day']:.2f}

PROFITABILITY
-------------
Initial Capital:       ${stats['initial_capital']:,.2f}
Current Capital:       ${stats['current_capital']:,.2f}
Peak Capital:          ${stats['peak_capital']:,.2f}
Total P&L:            ${stats['total_pnl']:,.2f}
ROI:                  {stats['roi']:.2f}%

Win Rate:             {stats['win_rate']:.2f}%
Winning Trades:       {stats['winning_trades']}
Losing Trades:        {stats['losing_trades']}

Average Win:          ${stats['avg_win']:,.2f}
Average Loss:         ${stats['avg_loss']:,.2f}
Average Trade:        ${stats['avg_trade']:,.2f}

Profit Factor:        {stats['profit_factor']:.2f}
Best Trade:           ${stats['best_trade']:,.2f}
Worst Trade:          ${stats['worst_trade']:,.2f}

RISK METRICS
------------
Sharpe Ratio:         {stats['sharpe_ratio']:.2f}
Sortino Ratio:        {stats['sortino_ratio']:.2f}

Max Drawdown:         {dd_info['max_drawdown_percent']:.2f}% (${dd_info['max_drawdown_value']:,.2f})
Current Drawdown:     {stats['current_drawdown_percent']:.2f}%
DD Duration:          {dd_info['drawdown_duration_days']} days
DD Recovery:          {dd_info['recovery_time_days']} days

STREAKS
-------
Max Winning Streak:   {stats['max_winning_streak']} trades
Max Losing Streak:    {stats['max_losing_streak']} trades
Current Streak:       {stats['current_streak']} trades

EFFICIENCY
----------
Avg Holding Period:   {stats['avg_holding_period_hours']:.1f} hours
{'='*70}
"""
        
        # Add monthly breakdown
        monthly_df = self.get_monthly_returns_summary()
        if not monthly_df.empty:
            report += "\nMONTHLY RETURNS\n"
            report += "-" * 70 + "\n"
            for _, row in monthly_df.iterrows():
                report += f"{row['month']:10} | P&L: ${row['pnl']:>10,.2f} | Return: {row['return_pct']:>6.2f}% | Trades: {int(row['trades']):>3}\n"
            report += "-" * 70 + "\n"
        
        # Add individual trades if requested
        if include_trades and self.trades:
            report += "\nRECENT TRADES (Last 10)\n"
            report += "-" * 70 + "\n"
            for trade in self.trades[-10:]:
                timestamp = trade.get('timestamp', 'N/A')
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime('%Y-%m-%d %H:%M')
                symbol = trade.get('symbol', 'N/A')
                side = trade.get('side', 'N/A')
                pnl = trade.get('pnl', 0)
                pnl_pct = trade.get('pnl_percent', 0)
                report += f"{timestamp} | {side:4} {symbol:10} | P&L: ${pnl:>8.2f} ({pnl_pct:>6.2f}%)\n"
        
        return report
    
    def export_to_json(self, filename: str):
        """Export all analytics data to JSON file"""
        data = {
            'stats': self.get_comprehensive_stats(),
            'drawdown_info': self.get_max_drawdown_info(),
            'monthly_returns': self.monthly_returns,
            'trade_distribution': self.get_trade_distribution(),
            'trades': self.trades,
            'equity_curve': [
                {
                    'timestamp': point['timestamp'].isoformat(),
                    'equity': point['equity'],
                    'drawdown': point['drawdown']
                }
                for point in self.equity_curve
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as pandas DataFrame for plotting"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve)
