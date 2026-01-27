import json
import os
from typing import List, Dict
from datetime import datetime

class VariantLogger:
    """
    Log variant data for training the difficulty estimator
    """
    
    def __init__(self, log_file: str = "data/variant_logs.json"):
        self.log_file = log_file
        self.logs: List[Dict] = []
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Load existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.logs = json.load(f)
    
    def log_variant(
        self,
        variant_text: str,
        root_problem: str,
        tree_depth: int,
        solve_success: bool,
        verification_time: float,
        reward: float,
        similarity_to_root: float = 0.0
    ):
        """Log a variant outcome"""
        log_entry = {
            'variant_text': variant_text,
            'root_problem': root_problem,
            'tree_depth': tree_depth,
            'solve_success': solve_success,
            'verification_time': verification_time,
            'reward': reward,
            'similarity_to_root': similarity_to_root,
            'variant_length': len(variant_text),
            'timestamp': datetime.now().isoformat()
        }
        self.logs.append(log_entry)
    
    def save(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def get_training_data(self) -> List[Dict]:
        """Get logs as training data"""
        return self.logs.copy()
    
    def get_stats(self) -> Dict:
        """Get statistics about logged variants"""
        if not self.logs:
            return {}
        
        total = len(self.logs)
        solved = sum(1 for log in self.logs if log['solve_success'])
        avg_verify_time = sum(log['verification_time'] for log in self.logs) / total
        avg_reward = sum(log['reward'] for log in self.logs) / total
        
        return {
            'total_variants': total,
            'solved_count': solved,
            'solve_rate': solved / total if total > 0 else 0,
            'avg_verification_time': avg_verify_time,
            'avg_reward': avg_reward
        }
