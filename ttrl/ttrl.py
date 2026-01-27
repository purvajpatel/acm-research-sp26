"""
TTRL: Test-Time Reinforcement Learning
Implements inference-time improvement as described in the paper
"""
import torch
from typing import List, Tuple, Optional
from core.variant_generator import VariantGenerator
from core.verifier import SolutionVerifier
from core.grpo_trainer import GRPOTrainer

class TTRL:
    """
    Test-Time Reinforcement Learning for inference-time model improvement
    As specified in the paper: lighter updates during inference
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        verifier: SolutionVerifier,
        variant_generator: VariantGenerator,
        learning_rate: float = 1e-6,  # Lighter than training (1e-5)
        num_variants: int = 5,  # As in paper
        num_iterations: int = 3,  # As in paper
        device: str = 'cuda'
    ):
        """
        Initialize TTRL
        
        Args:
            model: Language model to improve
            tokenizer: Model tokenizer
            verifier: Solution verifier
            variant_generator: Variant generator
            learning_rate: Learning rate for test-time updates (lighter than training)
            num_variants: Number of variants to generate (default 5 as in paper)
            num_iterations: Number of TTRL iterations (default 3 as in paper)
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.variant_generator = variant_generator
        self.learning_rate = learning_rate
        self.num_variants = num_variants
        self.num_iterations = num_iterations
        self.device = device
        
        # Create lightweight trainer for test-time updates
        self.trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=learning_rate,  # Lighter LR
            device=device
        )
    
    def solve_with_improvement(
        self,
        problem: str,
        initial_solution: Optional[str] = None
    ) -> Tuple[str, bool, List[float]]:
        """
        Solve problem with test-time improvement
        
        Args:
            problem: Problem to solve
            initial_solution: Optional initial solution (if already generated)
        
        Returns:
            Tuple of (best_solution, is_verified, accuracy_history)
        """
        # Generate initial solution if not provided
        if initial_solution is None:
            initial_solution = self._generate_solution(problem)
        
        # Verify initial solution
        is_verified, _ = self.verifier.verify_solution(problem, initial_solution)
        accuracy_history = [1.0 if is_verified else 0.0]
        
        best_solution = initial_solution
        best_verified = is_verified
        
        # TTRL iterations
        for iteration in range(self.num_iterations):
            # Generate variants
            variants = self.variant_generator.generate_simpler_variants(
                problem,
                num_variants=self.num_variants
            )
            
            # Solve variants
            variant_solutions = []
            verified_pairs = []
            
            for variant in variants:
                solution = self._generate_solution(variant)
                variant_solutions.append(solution)
                
                # Verify
                is_correct, _ = self.verifier.verify_solution(variant, solution)
                reward = 1.0 if is_correct else 0.0
                verified_pairs.append((variant, solution, reward))
            
            # Lightweight update on variants
            if len(verified_pairs) > 0:
                self.trainer.train_group(verified_pairs, batch_size=len(verified_pairs))
            
            # Try solving original problem again with improved model
            improved_solution = self._generate_solution(problem)
            is_verified, _ = self.verifier.verify_solution(problem, improved_solution)
            accuracy_history.append(1.0 if is_verified else 0.0)
            
            # Update best solution
            if is_verified and not best_verified:
                best_solution = improved_solution
                best_verified = True
            elif is_verified and best_verified:
                # Both verified, keep the newer one (or compare quality)
                best_solution = improved_solution
            elif not is_verified and not best_verified:
                # Neither verified, keep original
                pass
        
        return best_solution, best_verified, accuracy_history
    
    def _generate_solution(self, problem: str) -> str:
        """Generate solution for a problem"""
        prompt = f"""Solve the following integration problem step by step.

Problem: {problem}

Provide your solution in the format:
Solution: [antiderivative] + C

Solution:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract solution
        import re
        pattern = r'Solution:\s*(.+?)(?=\n\n|\nSolution:|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return response.split('\n')[-1].strip()
