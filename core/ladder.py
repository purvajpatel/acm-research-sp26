"""
LADDER: Learning through Autonomous Difficulty-Driven Example Recursion
Main framework implementation exactly as described in the paper
"""
from typing import List, Tuple, Optional, Dict
from .variant_tree import VariantTree, VariantNode
from .variant_generator import VariantGenerator
from .verifier import SolutionVerifier
from .grpo_trainer import GRPOTrainer
import torch

class LADDER:
    """
    LADDER framework: Recursive problem decomposition and self-improvement
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        verifier: SolutionVerifier,
        variant_generator: VariantGenerator,
        grpo_trainer: GRPOTrainer,
        max_depth: int = 5,
        num_variants: int = 3,
        device: str = 'cuda'
    ):
        """
        Initialize LADDER framework
        
        Args:
            model: Base language model
            tokenizer: Model tokenizer
            verifier: Solution verifier
            variant_generator: Variant generator
            grpo_trainer: GRPO trainer
            max_depth: Maximum recursion depth (default 5 as in paper)
            num_variants: Number of variants per level (default 3 as in paper)
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.variant_generator = variant_generator
        self.grpo_trainer = grpo_trainer
        self.max_depth = max_depth
        self.num_variants = num_variants
        self.device = device
    
    def process_problem(
        self,
        problem: str,
        solve_bottom_up: bool = True
    ) -> Tuple[VariantTree, Dict[str, str], Dict[str, bool]]:
        """
        Process a single problem through LADDER framework
        
        Args:
            problem: Initial problem to solve
            solve_bottom_up: Whether to solve from simplest to complex
        
        Returns:
            Tuple of (variant_tree, solutions_dict, verification_results)
        """
        # Step 1: Generate variant tree
        print(f"Generating variant tree for problem: {problem[:50]}...")
        variant_tree = self._generate_variant_tree(problem)
        
        # Step 2: Solve variants
        print(f"Solving {len(variant_tree)} variants...")
        solutions = {}
        verification_results = {}
        
        if solve_bottom_up:
            # Solve from simplest (leaves) to most complex (root)
            levels = variant_tree.get_levels()
            for level in reversed(levels):
                for node in level:
                    solution = self._solve_variant(node, solutions)
                    if solution:
                        solutions[node.id] = solution
                        # Verify solution
                        is_correct, reason = self.verifier.verify_solution(
                            node.problem, 
                            solution
                        )
                        verification_results[node.id] = is_correct
                        node.is_verified = is_correct
                        node.solution = solution
        else:
            # Solve all variants (for testing)
            for node in variant_tree.nodes:
                solution = self._solve_variant(node, solutions)
                if solution:
                    solutions[node.id] = solution
                    is_correct, reason = self.verifier.verify_solution(
                        node.problem,
                        solution
                    )
                    verification_results[node.id] = is_correct
                    node.is_verified = is_correct
                    node.solution = solution
        
        return variant_tree, solutions, verification_results
    
    def _generate_variant_tree(self, problem: str) -> VariantTree:
        """
        Generate variant tree recursively
        """
        tree = VariantTree()
        root = VariantNode(problem, depth=0)
        tree.add_node(root)
        
        queue = [root]
        
        while queue:
            current = queue.pop(0)
            
            if current.depth >= self.max_depth:
                continue
            
            # Generate simpler variants
            variant_problems = self.variant_generator.generate_simpler_variants(
                current.problem,
                num_variants=self.num_variants
            )
            
            for variant_problem in variant_problems:
                variant_node = VariantNode(
                    variant_problem,
                    depth=current.depth + 1,
                    parent=current
                )
                current.add_child(variant_node)
                tree.add_node(variant_node)
                
                if variant_node.depth < self.max_depth:
                    queue.append(variant_node)
        
        return tree
    
    def _solve_variant(
        self,
        node: VariantNode,
        existing_solutions: Dict[str, str]
    ) -> Optional[str]:
        """
        Solve a variant, optionally using child solutions as context
        """
        # If this is a leaf node, solve directly
        if node.is_leaf():
            return self._generate_solution(node.problem)
        
        # Otherwise, use child solutions as context
        child_solutions = []
        for child in node.children:
            if child.id in existing_solutions:
                child_solutions.append(
                    f"Problem: {child.problem}\nSolution: {existing_solutions[child.id]}"
                )
        
        if child_solutions:
            return self._generate_solution_with_context(
                node.problem,
                child_solutions
            )
        else:
            return self._generate_solution(node.problem)
    
    def _generate_solution(self, problem: str) -> Optional[str]:
        """Generate solution for a problem"""
        # Clear cache before generation
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        prompt = f"""Solve the following integration problem step by step.

Problem: {problem}

Provide your solution in the format:
Solution: [antiderivative] + C

Show your work:
1. Identify the integration technique
2. Apply the technique
3. Simplify the result

Solution:"""
        
        try:
            max_len = 256 if self.device == 'cpu' else 512
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_len
            ).to(self.device)
            
            with torch.no_grad():
                try:
                    max_tokens = 128 if self.device == 'cpu' else 256
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Clear cache and retry with smaller settings
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        raise
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clear memory
            del inputs, outputs
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Extract solution
            solution = self._extract_solution(response)
            return solution
        except Exception as e:
            print(f"Error generating solution: {e}")
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return None
    
    def _generate_solution_with_context(
        self,
        problem: str,
        child_solutions: List[str]
    ) -> Optional[str]:
        """Generate solution using child solutions as context"""
        context = "\n\n".join(child_solutions)
        
        prompt = f"""You have solved these similar problems:
{context}

Now solve this related but more complex problem:
Problem: {problem}

Use the techniques from the simpler problems as a guide.

Solution:"""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536  # Longer for context
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
            solution = self._extract_solution(response)
            return solution
        except Exception as e:
            print(f"Error generating solution with context: {e}")
            return None
    
    def _extract_solution(self, response: str) -> Optional[str]:
        """Extract solution from model response"""
        import re
        
        # Try to find "Solution:" pattern
        pattern = r'Solution:\s*(.+?)(?=\n\n|\nSolution:|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        
        if match:
            solution = match.group(1).strip()
            # Clean up
            solution = ' '.join(solution.split())
            return solution
        
        # Fallback: try to find last line that looks like a solution
        lines = response.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and ('∫' in line or 'integral' in line.lower() or '+' in line):
                return line
        
        return None
    
    def collect_verified_pairs(
        self,
        variant_tree: VariantTree,
        solutions: Dict[str, str],
        verification_results: Dict[str, bool]
    ) -> List[Tuple[str, str, float]]:
        """
        Collect verified problem-solution pairs for GRPO training
        
        Returns:
            List of (problem, solution, reward) tuples
            reward is 1.0 if verified correct, 0.0 otherwise
        """
        verified_pairs = []
        
        for node in variant_tree.nodes:
            if node.id in solutions and node.id in verification_results:
                reward = 1.0 if verification_results[node.id] else 0.0
                verified_pairs.append((node.problem, solutions[node.id], reward))
        
        return verified_pairs
    
    def train_on_problem(self, problem: str) -> Tuple[float, Dict]:
        """
        Process problem and train model on verified solutions
        
        Returns:
            Tuple of (average_loss, statistics)
        """
        # Process problem
        variant_tree, solutions, verification_results = self.process_problem(problem)
        
        # Collect verified pairs
        verified_pairs = self.collect_verified_pairs(
            variant_tree,
            solutions,
            verification_results
        )
        
        if len(verified_pairs) == 0:
            return 0.0, {'num_verified': 0, 'num_total': len(variant_tree)}
        
        # Train with GRPO
        loss = self.grpo_trainer.train_group(verified_pairs)
        
        stats = {
            'num_verified': sum(1 for _, _, r in verified_pairs if r > 0),
            'num_total': len(verified_pairs),
            'num_variants': len(variant_tree),
            'max_depth': variant_tree.max_depth
        }
        
        return loss, stats
