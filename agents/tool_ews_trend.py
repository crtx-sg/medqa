import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EWSTrendAnalyzer:
    """
    Comprehensive EWS trend analysis system combining multiple statistical and ML techniques
    for robust clinical trend detection and patient status assessment.
    """
    
    def __init__(self, window_size=6, recent_values_count=3, confidence_level=0.95):
        """
        Initialize the EWS Trend Analyzer
        
        Parameters:
        - window_size: Number of recent points for moving statistics
        - recent_values_count: Number of most recent values to assess immediate improvement
        - confidence_level: Statistical confidence level for trend significance
        """
        self.window_size = window_size
        self.recent_values_count = recent_values_count
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def preprocess_data(self, timestamps, ews_values):
        """
        Preprocess and validate input data
        """
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'ews': ews_values
        })
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates and handle missing values
        df = df.dropna().drop_duplicates(subset=['timestamp'])
        
        # Add time difference in hours for analysis
        if len(df) > 1:
            df['time_hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
        else:
            df['time_hours'] = [0]
            
        return df
    
    def linear_regression_trend(self, df):
        """
        Perform linear regression analysis for trend detection
        """
        if len(df) < 2:
            return {'slope': 0, 'r_squared': 0, 'p_value': 1.0, 'trend': 'insufficient_data'}
        
        X = df['time_hours'].values.reshape(-1, 1)
        y = df['ews'].values
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Calculate statistics
        y_pred = reg.predict(X)
        slope = reg.coef_[0]
        r_squared = reg.score(X, y)
        
        # Calculate p-value for slope significance
        n = len(df)
        if n > 2:
            # Standard error of slope
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (n - 2)
            se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X))**2))
            t_stat = slope / se_slope
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0
        
        # Determine trend direction
        if p_value < self.alpha:
            trend = 'deteriorating' if slope > 0 else 'improving'
        else:
            trend = 'stable'
            
        return {
            'slope': slope,
            'r_squared': r_squared,
            'p_value': p_value,
            'trend': trend,
            'significance': 'significant' if p_value < self.alpha else 'not_significant'
        }
    
    def mann_kendall_test(self, ews_values):
        """
        Perform Mann-Kendall test for monotonic trend detection
        """
        n = len(ews_values)
        if n < 3:
            return {'tau': 0, 'p_value': 1.0, 'trend': 'insufficient_data'}
        
        # Calculate S statistic
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if ews_values[j] > ews_values[i]:
                    S += 1
                elif ews_values[j] < ews_values[i]:
                    S -= 1
        
        # Calculate variance
        var_S = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if S > 0:
            Z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            Z = (S + 1) / np.sqrt(var_S)
        else:
            Z = 0
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        
        # Calculate Kendall's tau
        tau = S / (0.5 * n * (n - 1))
        
        # Determine trend
        if p_value < self.alpha:
            trend = 'deteriorating' if tau > 0 else 'improving'
        else:
            trend = 'stable'
            
        return {
            'tau': tau,
            'p_value': p_value,
            'trend': trend,
            'significance': 'significant' if p_value < self.alpha else 'not_significant'
        }
    
    def cusum_analysis(self, ews_values, target_mean=None, std_dev=None):
        """
        CUSUM analysis for change point detection
        """
        if target_mean is None:
            target_mean = np.mean(ews_values)
        if std_dev is None:
            std_dev = np.std(ews_values)
        
        if std_dev == 0:
            return {'cusum_pos': [0] * len(ews_values), 'cusum_neg': [0] * len(ews_values), 
                   'alerts': [], 'trend': 'stable'}
        
        # CUSUM parameters
        k = 0.5 * std_dev  # Reference value
        h = 4 * std_dev    # Control limit
        
        cusum_pos = [0]
        cusum_neg = [0]
        alerts = []
        
        for i, value in enumerate(ews_values[1:], 1):
            # Positive CUSUM (detecting upward shift - deterioration)
            cusum_pos_new = max(0, cusum_pos[-1] + (value - target_mean) - k)
            cusum_pos.append(cusum_pos_new)
            
            # Negative CUSUM (detecting downward shift - improvement)
            cusum_neg_new = min(0, cusum_neg[-1] + (value - target_mean) + k)
            cusum_neg.append(cusum_neg_new)
            
            # Check for alerts
            if cusum_pos_new > h:
                alerts.append({'index': i, 'type': 'deterioration', 'value': cusum_pos_new})
            elif cusum_neg_new < -h:
                alerts.append({'index': i, 'type': 'improvement', 'value': cusum_neg_new})
        
        # Determine overall trend
        recent_cusum_pos = cusum_pos[-min(self.window_size, len(cusum_pos)):]
        recent_cusum_neg = cusum_neg[-min(self.window_size, len(cusum_neg)):]
        
        if any(cp > h for cp in recent_cusum_pos):
            trend = 'deteriorating'
        elif any(cn < -h for cn in recent_cusum_neg):
            trend = 'improving'
        else:
            trend = 'stable'
            
        return {
            'cusum_pos': cusum_pos,
            'cusum_neg': cusum_neg,
            'alerts': alerts,
            'trend': trend
        }
    
    def exponential_smoothing_trend(self, ews_values, alpha=0.3):
        """
        Exponentially weighted moving average with trend detection
        """
        if len(ews_values) < 2:
            return {'ewma': ews_values, 'trend': 'insufficient_data', 'direction': None}
        
        ewma = [ews_values[0]]
        for i in range(1, len(ews_values)):
            ewma_new = alpha * ews_values[i] + (1 - alpha) * ewma[-1]
            ewma.append(ewma_new)
        
        # Calculate trend from EWMA
        recent_ewma = ewma[-min(self.window_size, len(ewma)):]
        if len(recent_ewma) >= 2:
            slope = (recent_ewma[-1] - recent_ewma[0]) / len(recent_ewma)
            
            # Determine significance based on change magnitude
            change_threshold = np.std(ews_values) * 0.1  # 10% of standard deviation
            
            if abs(slope) > change_threshold:
                trend = 'deteriorating' if slope > 0 else 'improving'
                direction = 'up' if slope > 0 else 'down'
            else:
                trend = 'stable'
                direction = 'stable'
        else:
            trend = 'stable'
            direction = 'stable'
            
        return {
            'ewma': ewma,
            'trend': trend,
            'direction': direction,
            'slope': slope if 'slope' in locals() else 0
        }
    
    def recent_values_assessment(self, ews_values):
        """
        Assess improvement based on most recent values
        """
        if len(ews_values) < self.recent_values_count:
            return {
                'improvement': None,
                'reason': 'insufficient_recent_data',
                'recent_slope': 0,
                'trend_strength': 'weak'
            }
        
        recent_values = ews_values[-self.recent_values_count:]
        
        # Simple linear trend of recent values
        x = np.arange(len(recent_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
        
        # Determine improvement status
        improvement = slope < 0  # Negative slope = improvement
        
        # Assess trend strength
        if abs(r_value) > 0.8:
            trend_strength = 'strong'
        elif abs(r_value) > 0.5:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        return {
            'improvement': improvement,
            'reason': f'slope_{"negative" if slope < 0 else "positive"}_last_{self.recent_values_count}_values',
            'recent_slope': slope,
            'trend_strength': trend_strength,
            'r_value': r_value,
            'p_value': p_value
        }
    
    def ensemble_analysis(self, df):
        """
        Combine multiple methods for robust trend analysis
        """
        ews_values = df['ews'].values
        
        # Individual analyses
        linear_result = self.linear_regression_trend(df)
        mk_result = self.mann_kendall_test(ews_values)
        cusum_result = self.cusum_analysis(ews_values)
        ewma_result = self.exponential_smoothing_trend(ews_values)
        recent_result = self.recent_values_assessment(ews_values)
        
        # Voting system for overall trend
        votes = []
        weights = []
        
        # Linear regression vote
        if linear_result['significance'] == 'significant':
            votes.append(linear_result['trend'])
            weights.append(linear_result['r_squared'])
        
        # Mann-Kendall vote
        if mk_result['significance'] == 'significant':
            votes.append(mk_result['trend'])
            weights.append(abs(mk_result['tau']))
        
        # CUSUM vote
        if cusum_result['trend'] != 'stable':
            votes.append(cusum_result['trend'])
            weights.append(0.8)  # Fixed weight for CUSUM
        
        # EWMA vote
        if ewma_result['trend'] != 'stable':
            votes.append(ewma_result['trend'])
            weights.append(0.6)  # Fixed weight for EWMA
        
        # Determine consensus
        if len(votes) == 0:
            overall_trend = 'stable'
            confidence = 'low'
        else:
            # Weighted voting
            improving_weight = sum(w for v, w in zip(votes, weights) if v == 'improving')
            deteriorating_weight = sum(w for v, w in zip(votes, weights) if v == 'deteriorating')
            
            if improving_weight > deteriorating_weight:
                overall_trend = 'improving'
            elif deteriorating_weight > improving_weight:
                overall_trend = 'deteriorating'
            else:
                overall_trend = 'stable'
            
            # Calculate confidence
            total_weight = improving_weight + deteriorating_weight
            max_weight = max(improving_weight, deteriorating_weight)
            confidence_ratio = max_weight / total_weight if total_weight > 0 else 0
            
            if confidence_ratio > 0.7:
                confidence = 'high'
            elif confidence_ratio > 0.5:
                confidence = 'moderate'
            else:
                confidence = 'low'
        
        return {
            'overall_trend': overall_trend,
            'confidence': confidence,
            'improvement': recent_result['improvement'],
            'individual_results': {
                'linear_regression': linear_result,
                'mann_kendall': mk_result,
                'cusum': cusum_result,
                'ewma': ewma_result,
                'recent_assessment': recent_result
            },
            'voting_summary': {
                'votes': votes,
                'weights': weights,
                'improving_weight': improving_weight if 'improving_weight' in locals() else 0,
                'deteriorating_weight': deteriorating_weight if 'deteriorating_weight' in locals() else 0
            }
        }
    
    def analyze_trend(self, timestamps, ews_values):
        """
        Main method to analyze EWS trend
        """
        # Preprocess data
        df = self.preprocess_data(timestamps, ews_values)
        
        if len(df) < 2:
            return {
                'status': 'insufficient_data',
                'improvement': None,
                'message': 'Need at least 2 data points for trend analysis'
            }
        
        # Perform ensemble analysis
        results = self.ensemble_analysis(df)
        
        # Generate summary
        summary = {
            'patient_status': results['overall_trend'],
            'improvement': results['improvement'],
            'confidence': results['confidence'],
            'data_points': len(df),
            'time_span_hours': df['time_hours'].iloc[-1] if len(df) > 1 else 0,
            'current_ews': df['ews'].iloc[-1],
            'trend_details': results['individual_results'],
            'clinical_interpretation': self._generate_clinical_interpretation(results)
        }
        
        return summary
    
    def _generate_clinical_interpretation(self, results):
        """
        Generate clinical interpretation of results
        """
        trend = results['overall_trend']
        confidence = results['confidence']
        improvement = results['improvement']
        
        interpretation = {
            'primary_finding': '',
            'clinical_action': '',
            'monitoring_recommendation': ''
        }
        
        if trend == 'improving':
            interpretation['primary_finding'] = f"Patient shows improving trend (confidence: {confidence})"
            interpretation['clinical_action'] = "Continue current management"
            interpretation['monitoring_recommendation'] = "Regular monitoring, consider reducing frequency if trend continues"
        elif trend == 'deteriorating':
            interpretation['primary_finding'] = f"Patient shows deteriorating trend (confidence: {confidence})"
            interpretation['clinical_action'] = "Consider escalation of care and intervention"
            interpretation['monitoring_recommendation'] = "Increase monitoring frequency, review treatment plan"
        else:
            interpretation['primary_finding'] = f"Patient condition appears stable (confidence: {confidence})"
            interpretation['clinical_action'] = "Continue current monitoring"
            interpretation['monitoring_recommendation'] = "Maintain current monitoring schedule"
        
        # Recent trend assessment
        if improvement is not None:
            recent_status = "improving" if improvement else "deteriorating"
            interpretation['recent_trend'] = f"Most recent values suggest {recent_status} trajectory"
        else:
            interpretation['recent_trend'] = "Insufficient recent data for short-term assessment"
        
        return interpretation
    
    def plot_analysis(self, timestamps, ews_values, save_path=None):
        """
        Create visualization of EWS trend analysis
        """
        df = self.preprocess_data(timestamps, ews_values)
        
        if len(df) < 2:
            print("Insufficient data for plotting")
            return
        
        # Perform analysis
        results = self.analyze_trend(timestamps, ews_values)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: EWS over time with linear trend
        ax1.plot(df['timestamp'], df['ews'], 'bo-', label='EWS Values', markersize=6)
        
        # Add linear regression line
        linear_result = results['trend_details']['linear_regression']
        if len(df) > 1:
            X = df['time_hours'].values.reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(X, df['ews'].values)
            trend_line = reg.predict(X)
            ax1.plot(df['timestamp'], trend_line, 'r--', 
                    label=f'Linear Trend (slope={linear_result["slope"]:.3f})', linewidth=2)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('EWS Score')
        ax1.set_title('EWS Trend Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: EWMA
        ewma_result = results['trend_details']['ewma']
        ax2.plot(df['timestamp'], df['ews'], 'bo-', label='EWS Values', alpha=0.6)
        ax2.plot(df['timestamp'], ewma_result['ewma'], 'g-', label='EWMA', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('EWS Score')
        ax2.set_title('Exponentially Weighted Moving Average')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: CUSUM
        cusum_result = results['trend_details']['cusum']
        ax3.plot(range(len(cusum_result['cusum_pos'])), cusum_result['cusum_pos'], 
                'r-', label='CUSUM+', linewidth=2)
        ax3.plot(range(len(cusum_result['cusum_neg'])), cusum_result['cusum_neg'], 
                'b-', label='CUSUM-', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Time Point')
        ax3.set_ylabel('CUSUM Value')
        ax3.set_title('CUSUM Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        methods = ['Linear\nRegression', 'Mann-Kendall', 'CUSUM', 'EWMA']
        trends = [
            results['trend_details']['linear_regression']['trend'],
            results['trend_details']['mann_kendall']['trend'],
            results['trend_details']['cusum']['trend'],
            results['trend_details']['ewma']['trend']
        ]
        
        # Convert trends to numeric for plotting
        trend_map = {'improving': -1, 'stable': 0, 'deteriorating': 1}
        trend_values = [trend_map.get(t, 0) for t in trends]
        
        colors = ['green' if tv == -1 else 'red' if tv == 1 else 'orange' for tv in trend_values]
        
        bars = ax4.bar(methods, trend_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Trend Direction')
        ax4.set_title('Method Consensus')
        ax4.set_ylim(-1.5, 1.5)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(['Improving', 'Stable', 'Deteriorating'])
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, trend in zip(bars, trends):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1 if height >= 0 else height - 0.1,
                    trend, ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary
        print(f"\n=== EWS TREND ANALYSIS SUMMARY ===")
        print(f"Overall Trend: {results['patient_status'].upper()}")
        print(f"Recent Improvement: {results['improvement']}")
        print(f"Confidence: {results['confidence'].upper()}")
        print(f"Data Points: {results['data_points']}")
        print(f"Time Span: {results['time_span_hours']:.1f} hours")
        print(f"Current EWS: {results['current_ews']}")
        print(f"\nClinical Interpretation:")
        for key, value in results['clinical_interpretation'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")


# Example usage and demonstration
def demo_ews_analysis():
    """
    Demonstrate the EWS trend analysis system with sample data
    """
    print("=== EWS TREND ANALYSIS DEMONSTRATION ===\n")
    
    # Initialize analyzer
    analyzer = EWSTrendAnalyzer(window_size=6, recent_values_count=3)
    
    # Sample data - deteriorating patient
    timestamps_deteriorating = [
        datetime.now() - timedelta(hours=12),
        datetime.now() - timedelta(hours=10),
        datetime.now() - timedelta(hours=8),
        datetime.now() - timedelta(hours=6),
        datetime.now() - timedelta(hours=4),
        datetime.now() - timedelta(hours=2),
        datetime.now()
    ]
    ews_deteriorating = [2, 3, 4, 5, 7, 8, 9]
    
    print("1. DETERIORATING PATIENT SCENARIO:")
    print(f"EWS Values: {ews_deteriorating}")
    
    results_deteriorating = analyzer.analyze_trend(timestamps_deteriorating, ews_deteriorating)
    print(f"Overall Trend: {results_deteriorating['patient_status']}")
    print(f"Recent Improvement: {results_deteriorating['improvement']}")
    print(f"Confidence: {results_deteriorating['confidence']}")
    
    # Sample data - improving patient
    timestamps_improving = [
        datetime.now() - timedelta(hours=12),
        datetime.now() - timedelta(hours=10),
        datetime.now() - timedelta(hours=8),
        datetime.now() - timedelta(hours=6),
        datetime.now() - timedelta(hours=4),
        datetime.now() - timedelta(hours=2),
        datetime.now()
    ]
    ews_improving = [9, 7, 6, 4, 3, 2, 1]
    
    print(f"\n2. IMPROVING PATIENT SCENARIO:")
    print(f"EWS Values: {ews_improving}")
    
    results_improving = analyzer.analyze_trend(timestamps_improving, ews_improving)
    print(f"Overall Trend: {results_improving['patient_status']}")
    print(f"Recent Improvement: {results_improving['improvement']}")
    print(f"Confidence: {results_improving['confidence']}")
    
    # Sample data - stable patient
    timestamps_stable = [
        datetime.now() - timedelta(hours=12),
        datetime.now() - timedelta(hours=10),
        datetime.now() - timedelta(hours=8),
        datetime.now() - timedelta(hours=6),
        datetime.now() - timedelta(hours=4),
        datetime.now() - timedelta(hours=2),
        datetime.now()
    ]
    ews_stable = [4, 3, 4, 4, 3, 4, 3]
    
    print(f"\n3. STABLE PATIENT SCENARIO:")
    print(f"EWS Values: {ews_stable}")
    
    results_stable = analyzer.analyze_trend(timestamps_stable, ews_stable)
    print(f"Overall Trend: {results_stable['patient_status']}")
    print(f"Recent Improvement: {results_stable['improvement']}")
    print(f"Confidence: {results_stable['confidence']}")
    
    return analyzer, (timestamps_deteriorating, ews_deteriorating)

# Run demonstration
if __name__ == "__main__":
    analyzer, sample_data = demo_ews_analysis()
    
    # Create visualization for the deteriorating patient
    print(f"\nGenerating visualization for deteriorating patient scenario...")
    analyzer.plot_analysis(sample_data[0], sample_data[1])
