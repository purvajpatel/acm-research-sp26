"""
LADDER with Adaptive Controller (ASCEND)
Integrates adaptive filtering into the LADDER framework
"""
from typing import List, Tuple, Optional, Dict
from .variant_tree import VariantTree, VariantNode
from .variant_generator import VariantGenerator
from .verifier import SolutionVerifier
from .grpo_trainer import GRPOTrainer
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from controllers.adaptive_controller import AdaptiveController
import torch

class LADDERAdaptive:
    """
    LADDER framework with adaptive variant filtering (ASCEND)
    Uses difficulty estimator to filter wasteful variants
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        verifier: SolutionVerifier,
        variant_generator: VariantGenerator,
        grpo_trainer: GRPOTrainer,
        adaptive_controller: AdaptiveController,
        max_depth: int = 5,
        num_variants: int = 3,
        device: str = 'cuda'
    ):
        """
        Initialize Adaptive LADDER framework
        
        Args:
            model: Base language model
            tokenizer: Model tokenizer
            verifier: Solution verifier
            variant_generator: Variant generator
            grpo_trainer: GRPO trainer
            adaptive_controller: Adaptive controller for filtering
            max_depth: Maximum recursion depth
            num_variants: Number of variants per level (before filtering)
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.variant_generator = variant_generator
        self.grpo_trainer = grpo_trainer
        self.adaptive_controller = adaptive_controller
        self.max_depth = max_depth
        self.num_variants = num_variants
        self.device = device
    
    def process_problem(
        self,
        problem: str,
        solve_bottom_up: bool = True
    ) -> Tuple[VariantTree, Dict[str, str], Dict[str, bool]]:
        """
        Process problem with adaptive filtering
        """
        # Reset controller stats for new problem
        self.adaptive_controller.reset_stats()
        
        # Generate variant tree with adaptive filtering
        variant_tree = self._generate_variant_tree_adaptive(problem)
        
        # Solve variants (same as baseline)
        solutions = {}
        verification_results = {}
        
        if solve_bottom_up:
            # Solve from leaves to root
            for depth in range(variant_tree.max_depth, -1, -1):
                nodes_at_depth = [n for n in variant_tree.nodes if n.depth == depth]
                for node in nodes_at_depth:
                    solution = self._solve_variant(node, solutions)
                    if solution:
                        solutions[node.id] = solution
                        is_correct, _ = self.verifier.verify_solution(node.problem, solution)
                        verification_results[node.id] = is_correct
                        node.is_verified = is_correct
                        node.solution = solution
        else:
            # Solve in order
            for node in variant_tree.nodes:
                solution = self._solve_variant(node, solutions)
                if solution:
                    solutions[node.id] = solution
                    is_correct, _ = self.verifier.verify_solution(node.problem, solution)
                    verification_results[node.id] = is_correct
                    node.is_verified = is_correct
                    node.solution = solution
        
        return variant_tree, solutions, verification_results
    
    def _generate_variant_tree_adaptive(self, problem: str) -> VariantTree:
        """
        Generate variant tree with adaptive filtering
        """
        tree = VariantTree()
        root = VariantNode(problem, depth=0)
        tree.add_node(root)
        
        queue = [root]
        
        while queue:
            current = queue.pop(0)
            
            if current.depth >= self.max_depth:
                continue
            
            # Generate variants
            variant_problems = self.variant_generator.generate_simpler_variants(
                current.problem,
                num_variants=self.num_variants
            )
            
            # Filter variants using adaptive controller
            filtered_variants = []
            for variant_problem in variant_problems:
                self.adaptive_controller.stats['variants_generated'] += 1
                
                # Compute similarity to root
                similarity = self._compute_similarity(variant_problem, problem)
                
                # Check if variant should be accepted
                should_accept, predictions = self.adaptive_controller.should_accept_variant(
                    variant_text=variant_problem,
                    depth=current.depth + 1,
                    similarity_to_root=similarity,
                    root_problem=problem
                )
                
                if should_accept:
                    filtered_variants.append(variant_problem)
            
            # Add filtered variants to tree
            for variant_problem in filtered_variants:
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
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity metric"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def _solve_variant(
        self,
        node: VariantNode,
        existing_solutions: Dict[str, str]
    ) -> Optional[str]:
        """Solve a variant (same as baseline LADDER)"""
        if node.children:
            # Use child solutions as context
            context = "\n".join([
                f"Problem: {child.problem}\nSolution: {existing_solutions.get(child.id, '')}"
                for child in node.children
                if child.id in existing_solutions
            ])
            return self._generate_solution_with_context(node.problem, context)
        else:
            return self._generate_solution(node.problem)
    
    def _generate_solution(self, problem: str) -> Optional[str]:
        """Generate solution for a problem"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        prompt = f"""Solve the following integration problem step by step.

Problem: {problem}

Provide your solution in the format:
Solution: [antiderivative] + C

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
                max_tokens = 128 if self.device == 'cpu' else 256
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            del inputs, outputs
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return self._extract_solution(response)
        except Exception as e:
            print(f"Error generating solution: {e}")
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return None
    
    def _generate_solution_with_context(self, problem: str, context: str) -> Optional[str]:
        """Generate solution with context from child solutions"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        prompt = f"""You have solved these simpler problems:
{context}

Now solve this harder problem using similar techniques:

Problem: {problem}

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
                max_tokens = 128 if self.device == 'cpu' else 256
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            del inputs, outputs
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return self._extract_solution(response)
        except Exception as e:
            print(f"Error generating solution with context: {e}")
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return None
    
    def _extract_solution(self, response: str) -> Optional[str]:
        """Extract solution from model response"""
        import re
        pattern = r'Solution:\s*(.+?)(?=\n\n|\nSolution:|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.split('\n')[-1].strip() if response else None
    
    def collect_verified_pairs(
        self,
        variant_tree: VariantTree,
        solutions: Dict[str, str],
        verification_results: Dict[str, bool]
    ) -> List[Tuple[str, str, float]]:
        """Collect verified problem-solution pairs"""
        verified_pairs = []
        for node in variant_tree.nodes:
            if node.id in solutions and node.id in verification_results:
                reward = 1.0 if verification_results[node.id] else 0.0
                verified_pairs.append((node.problem, solutions[node.id], reward))
        return verified_pairs
    
    def train_on_problem(self, problem: str) -> Tuple[float, Dict]:
        """Process problem and train model on verified solutions"""
        variant_tree, solutions, verification_results = self.process_problem(problem)
        verified_pairs = self.collect_verified_pairs(variant_tree, solutions, verification_results)
        
        if len(verified_pairs) == 0:
            return 0.0, {'num_verified': 0, 'num_total': len(variant_tree)}
        
        loss = self.grpo_trainer.train_group(verified_pairs)
        
        stats = {
            'num_verified': sum(1 for _, _, r in verified_pairs if r > 0),
            'num_total': len(verified_pairs),
            'num_variants': len(variant_tree),
            'max_depth': variant_tree.max_depth,
            'adaptive_stats': self.adaptive_controller.get_stats()
        }
        
        return loss, stats
