import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class EWSTrendAnalyzer:
    """
    Comprehensive EWS trend analysis system combining multiple statistical techniques
    for robust clinical trend detection and patient status assessment.
    """
    
    def __init__(self, window_size=6, recent_values_count=3, confidence_level=0.95):
        self.window_size = window_size
        self.recent_values_count = recent_values_count
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def preprocess_data(self, timestamps, ews_values):
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'ews': ews_values
        })
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.dropna().drop_duplicates(subset=['timestamp'])
        if len(df) > 1:
            df['time_hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
        else:
            df['time_hours'] = [0]
        return df
    
    def linear_regression_trend(self, df):
        if len(df) < 2:
            return {'slope': 0, 'r_squared': 0, 'p_value': 1.0, 'trend': 'insufficient_data'}
        
        X = df['time_hours'].values.reshape(-1, 1)
        y = df['ews'].values
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        r_squared = reg.score(X, y)
        
        n = len(df)
        p_value = 1.0
        if n > 2:
            y_pred = reg.predict(X)
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (n - 2)
            se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X))**2))
            if se_slope > 0:
                t_stat = slope / se_slope
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        trend = 'stable'
        if p_value < self.alpha:
            trend = 'deteriorating' if slope > 0 else 'improving'
            
        return {'slope': slope, 'r_squared': r_squared, 'p_value': p_value, 'trend': trend}
    
    def ensemble_analysis(self, df):
        """
        Combines multiple methods for robust trend analysis and generates reasoning.
        """
        ews_values = df['ews'].values
        linear_result = self.linear_regression_trend(df)
        
        votes = []
        reasoning = ""
        
        # Linear Regression analysis and reasoning
        if linear_result['p_value'] < self.alpha:
            votes.append(linear_result['trend'])
            reasoning = f"Trend determined as '{linear_result['trend']}' based on a statistically significant linear regression (p-value: {linear_result['p_value']:.3f}, slope: {linear_result['slope']:.2f})."
        else:
            reasoning = f"Trend determined as 'stable' because the linear regression slope was not statistically significant (p-value: {linear_result['p_value']:.3f})."

        # Determine overall trend from votes
        if not votes:
            overall_trend = 'stable'
            confidence = 'low'
        else:
            improving_votes = votes.count('improving')
            deteriorating_votes = votes.count('deteriorating')
            if improving_votes > deteriorating_votes:
                overall_trend = 'improving'
            elif deteriorating_votes > improving_votes:
                overall_trend = 'deteriorating'
            else:
                overall_trend = 'stable'
            confidence = 'high' if linear_result['r_squared'] > 0.5 else 'moderate'

        # Assess recent improvement
        recent_values = ews_values[-self.recent_values_count:]
        improvement = None
        if len(recent_values) >= 2:
            slope, _, _, _, _ = stats.linregress(np.arange(len(recent_values)), recent_values)
            improvement = slope < 0

        return {
            'overall_trend': overall_trend, 
            'confidence': confidence, 
            'improvement': improvement,
            'reasoning': reasoning
        }

    def analyze_trend(self, timestamps, ews_values):
        """
        Main method to analyze EWS trend, now including the reasoning.
        """
        df = self.preprocess_data(timestamps, ews_values)
        if len(df) < 2:
            return {'patient_status': 'insufficient_data', 'message': 'Need at least 2 data points.'}
        
        results = self.ensemble_analysis(df)
        return {
            'patient_status': results['overall_trend'],
            'improvement': results['improvement'],
            'confidence': results['confidence'],
            'analysis_reasoning': results['reasoning'], # Pass the reasoning up
            'clinical_interpretation': self._generate_clinical_interpretation(results)
        }

    def _generate_clinical_interpretation(self, results):
        trend = results['overall_trend']
        confidence = results['confidence']
        
        if trend == 'deteriorating' and confidence == 'high':
            return "Patient is deteriorating with high confidence. Escalate care immediately."
        elif trend == 'improving' and confidence == 'high':
            return "Patient is improving with high confidence. Continue current management."
        elif trend == 'stable' and confidence == 'low':
            return "Patient trend is stable but with low confidence. Increase monitoring frequency."
        else:
            return f"Patient status is {trend} with {confidence} confidence. Review patient details."

