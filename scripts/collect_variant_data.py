"""
Collect variant data for training the difficulty estimator
Runs LADDER on small dataset and logs all variant outcomes
"""
import argparse
import torch
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_loader import ModelLoader
from core.verifier import SolutionVerifier
from core.variant_generator import VariantGenerator
from core.grpo_trainer import GRPOTrainer
from core.ladder import LADDER
from data.dataset_loader import IntegrationDataset
from utils.variant_logger import VariantLogger

def compute_similarity(text1: str, text2: str) -> float:
    """Simple similarity metric (can be improved)"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0

def main():
    parser = argparse.ArgumentParser(description='Collect variant data for estimator training')
    parser.add_argument('--model', type=str, default='llama3.2-3b',
                       choices=['llama3.2-3b', 'qwen2.5-7b', 'deepseek-r1-7b'])
    parser.add_argument('--dataset', type=str, default='undergraduate',
                       choices=['undergraduate', 'mit_bee'])
    parser.add_argument('--max_problems', type=int, default=10,
                       help='Number of problems to process')
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--num_variants', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='data/variant_logs.json')
    parser.add_argument('--hf_token', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    print("⚠ This is for data collection - CPU is fine for small scale")
    
    # Initialize logger
    logger = VariantLogger(args.output)
    
    # Load model
    print(f"Loading {args.model}...")
    model, tokenizer = ModelLoader.load_model(
        model_name=args.model,
        device=device,
        use_auth_token=args.hf_token or os.getenv('HUGGINGFACE_TOKEN')
    )
    
    # Load dataset
    dataset_loader = IntegrationDataset()
    if args.dataset == 'undergraduate':
        problems = dataset_loader.get_undergraduate_problems()[:args.max_problems]
    else:
        problems = dataset_loader.get_mit_bee_problems()[:args.max_problems]
    
    print(f"Processing {len(problems)} problems for data collection...")
    
    # Initialize components
    verifier = SolutionVerifier(tolerance=1e-6, num_test_points=50)
    variant_generator = VariantGenerator(model, tokenizer, device=device)
    grpo_trainer = GRPOTrainer(model=model, tokenizer=tokenizer, learning_rate=1e-5, device=device)
    
    # Modified LADDER that logs variants
    ladder = LADDER(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        variant_generator=variant_generator,
        grpo_trainer=grpo_trainer,
        max_depth=args.max_depth,
        num_variants=args.num_variants,
        device=device
    )
    
    # Process problems and log variants
    import time
    for i, problem in enumerate(problems):
        print(f"\nProcessing problem {i+1}/{len(problems)}: {problem[:50]}...")
        
        # Process problem
        variant_tree, solutions, verification_results = ladder.process_problem(problem)
        
        # Log all variants
        for node in variant_tree.nodes:
            variant_text = node.problem
            depth = node.depth
            solve_success = verification_results.get(node.id, False)
            
            # Measure verification time
            start_time = time.time()
            is_correct, _ = verifier.verify_solution(variant_text, solutions.get(node.id, ""))
            verify_time = time.time() - start_time
            
            reward = 1.0 if solve_success else 0.0
            similarity = compute_similarity(variant_text, problem)
            
            logger.log_variant(
                variant_text=variant_text,
                root_problem=problem,
                tree_depth=depth,
                solve_success=solve_success,
                verification_time=verify_time,
                reward=reward,
                similarity_to_root=similarity
            )
        
        # Save periodically
        if (i + 1) % 5 == 0:
            logger.save()
            stats = logger.get_stats()
            print(f"  Logged {stats['total_variants']} variants so far")
    
    # Final save
    logger.save()
    stats = logger.get_stats()
    
    print(f"\n✓ Data collection complete!")
    print(f"  Total variants logged: {stats['total_variants']}")
    print(f"  Solve rate: {stats['solve_rate']:.2%}")
    print(f"  Avg verification time: {stats['avg_verification_time']:.3f}s")
    print(f"  Logs saved to: {args.output}")

if __name__ == '__main__':
    main()
