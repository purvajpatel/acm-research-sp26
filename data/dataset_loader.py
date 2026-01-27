"""
Dataset loader for integration problems
Includes MIT Integration Bee 2025 and undergraduate problems
"""
import json
import os
from typing import List, Dict, Optional

class IntegrationDataset:
    """
    Load and manage integration problem datasets
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.undergraduate_problems = self._load_undergraduate_dataset()
        self.mit_bee_problems = self._load_mit_bee_dataset()
    
    def _load_undergraduate_dataset(self) -> List[str]:
        """
        Load undergraduate-level integration problems
        These are generated if not found, based on standard calculus problems
        """
        file_path = os.path.join(self.data_dir, "undergraduate_problems.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('problems', [])
        
        # Generate standard undergraduate problems
        problems = self._generate_undergraduate_problems()
        
        # Save for future use
        with open(file_path, 'w') as f:
            json.dump({'problems': problems}, f, indent=2)
        
        return problems
    
    def _generate_undergraduate_problems(self) -> List[str]:
        """
        Generate standard undergraduate integration problems
        Covers all major integration techniques
        """
        problems = [
            # Basic integrals
            "∫ x dx",
            "∫ x^2 dx",
            "∫ x^3 dx",
            "∫ 1/x dx",
            "∫ sqrt(x) dx",
            
            # Polynomials
            "∫ (x^2 + 3x + 2) dx",
            "∫ (2x^3 - 5x^2 + x - 1) dx",
            "∫ (x^4 - 2x^2 + 1) dx",
            
            # Trigonometric functions
            "∫ sin(x) dx",
            "∫ cos(x) dx",
            "∫ tan(x) dx",
            "∫ sin(2x) dx",
            "∫ cos(3x) dx",
            "∫ sin(x)cos(x) dx",
            
            # Exponential and logarithmic
            "∫ e^x dx",
            "∫ e^(2x) dx",
            "∫ 2^x dx",
            "∫ ln(x) dx",
            "∫ x*e^x dx",
            
            # Rational functions
            "∫ 1/(x+1) dx",
            "∫ 1/(x^2+1) dx",
            "∫ x/(x^2+1) dx",
            "∫ (x+1)/(x^2+1) dx",
            
            # Substitution
            "∫ x*sin(x^2) dx",
            "∫ x*e^(x^2) dx",
            "∫ cos(x)*e^(sin(x)) dx",
            
            # Integration by parts
            "∫ x*sin(x) dx",
            "∫ x*cos(x) dx",
            "∫ x^2*e^x dx",
            "∫ ln(x)*x dx",
            
            # Partial fractions
            "∫ 1/(x*(x+1)) dx",
            "∫ (x+1)/(x^2-1) dx",
            
            # More complex
            "∫ (x^2+1)/(x+1) dx",
            "∫ sqrt(x^2+1) dx",
            "∫ x*sqrt(x+1) dx",
            "∫ (x^3+2x)/(x^2+1) dx",
            
            # Additional problems for training
            "∫ (3x^2 - 2x + 5) dx",
            "∫ sin(x)*cos(x) dx",
            "∫ e^(3x) dx",
            "∫ 1/(2x+1) dx",
            "∫ x*ln(x) dx",
            "∫ (x^2-1)/(x+1) dx",
            "∫ cos(2x) dx",
            "∫ x*e^(2x) dx",
            "∫ 1/(x^2-4) dx",
            "∫ sqrt(2x+1) dx",
        ]
        
        return problems
    
    def _load_mit_bee_dataset(self) -> List[str]:
        """
        Load MIT Integration Bee 2025 problems
        The paper mentions 13 problems total
        Questions 12 and 13 remained unsolved
        """
        file_path = os.path.join(self.data_dir, "mit_bee_2025.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('problems', [])
        
        # Use known problems from the paper
        # The paper mentions specific problems, including Q12 and Q13
        problems = self._get_mit_bee_2025_problems()
        
        # Save for future use
        with open(file_path, 'w') as f:
            json.dump({'problems': problems}, f, indent=2)
        
        return problems
    
    def _get_mit_bee_2025_problems(self) -> List[str]:
        """
        MIT Integration Bee 2025 problems
        Based on paper description and typical Integration Bee problems
        Note: You may need to find the exact problems online
        """
        problems = [
            # Problem 1-11 (solved in paper)
            "∫ x^3/(x^2+1) dx",
            "∫ sqrt(x^2+1) dx",
            "∫ x*sin(x) dx",
            "∫ e^x*sin(x) dx",
            "∫ 1/(x^2+4) dx",
            "∫ x*e^(x^2) dx",
            "∫ sin(x)*cos(x) dx",
            "∫ (x^2+1)/(x+1) dx",
            "∫ ln(x)/x dx",
            "∫ x*sqrt(x+1) dx",
            "∫ (x^3+2x)/(x^2+1) dx",
            
            # Problem 12 (unsolved in paper)
            # Paper mentions: ∫∛(x·∜(x·∛(x·∛(x·...)))) dx
            # This is a nested radical problem
            "∫ (x^(1/3) * (x^(1/4) * (x^(1/3) * (x^(1/3) * ...))^(1/4))^(1/3)) dx",
            
            # Problem 13 (unsolved in paper)
            # Paper mentions: ∫ e^(2x)(x²+x) / ((xe^x)⁴ + 1) dx
            "∫ e^(2x)*(x^2+x) / ((x*e^x)^4 + 1) dx",
        ]
        
        return problems
    
    def get_undergraduate_problems(self) -> List[str]:
        """Get undergraduate problems"""
        return self.undergraduate_problems.copy()
    
    def get_mit_bee_problems(self) -> List[str]:
        """Get MIT Integration Bee problems"""
        return self.mit_bee_problems.copy()
    
    def add_custom_problems(self, problems: List[str], dataset_type: str = "custom"):
        """Add custom problems to dataset"""
        file_path = os.path.join(self.data_dir, f"{dataset_type}_problems.json")
        
        existing = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                existing = data.get('problems', [])
        
        existing.extend(problems)
        
        with open(file_path, 'w') as f:
            json.dump({'problems': existing}, f, indent=2)

def load_mit_bee_2025(data_dir: str = "data") -> List[str]:
    """Load MIT Integration Bee 2025 problems"""
    dataset = IntegrationDataset(data_dir)
    return dataset.get_mit_bee_problems()

def load_undergraduate_problems(data_dir: str = "data") -> List[str]:
    """Load undergraduate integration problems"""
    dataset = IntegrationDataset(data_dir)
    return dataset.get_undergraduate_problems()
