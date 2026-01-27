"""
Compare baseline LADDER vs Adaptive ASCEND
Shows efficiency improvements without full training
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
from core.ladder_adaptive import LADDERAdaptive
from data.dataset_loader import IntegrationDataset
from controllers.adaptive_controller import AdaptiveController
from estimators.difficulty_estimator import DifficultyEstimator
import time

def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs adaptive')
    parser.add_argument('--model', type=str, default='llama3.2-3b')
    parser.add_argument('--dataset', type=str, default='undergraduate')
    parser.add_argument('--max_problems', type=int, default=5)
    parser.add_argument('--estimator_path', type=str, default='checkpoints/estimator.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hf_token', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    # Load model
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
    
    print(f"Comparing on {len(problems)} problems...")
    
    # Initialize components
    verifier = SolutionVerifier()
    variant_generator = VariantGenerator(model, tokenizer, device=device)
    grpo_trainer = GRPOTrainer(model=model, tokenizer=tokenizer, device=device)
    
    # Load estimator
    estimator = DifficultyEstimator()
    if os.path.exists(args.estimator_path):
        estimator.load_state_dict(torch.load(args.estimator_path, map_location='cpu'))
        print("✓ Loaded trained estimator")
    else:
        print("⚠ Estimator not found. Using untrained estimator (for demo)")
    
    # Initialize adaptive controller
    adaptive_controller = AdaptiveController(estimator)
    
    # Baseline: Standard LADDER
    print("\n" + "=" * 60)
    print("BASELINE LADDER (All Variants)")
    print("=" * 60)
    
    baseline_ladder = LADDER(
        model=model, tokenizer=tokenizer, verifier=verifier,
        variant_generator=variant_generator, grpo_trainer=grpo_trainer,
        max_depth=3, num_variants=3, device=device
    )
    
    baseline_stats = {
        'total_variants': 0,
        'compute_time': 0.0,
        'solved': 0,
        'total_problems': len(problems)
    }
    
    print("Running baseline LADDER...")
    for i, problem in enumerate(problems):
        print(f"  Problem {i+1}/{len(problems)}...")
        start_time = time.time()
        variant_tree, solutions, verification_results = baseline_ladder.process_problem(problem)
        elapsed = time.time() - start_time
        
        baseline_stats['total_variants'] += len(variant_tree)
        baseline_stats['compute_time'] += elapsed
        baseline_stats['solved'] += sum(1 for v in verification_results.values() if v)
    
    # Adaptive: ASCEND with filtering
    print("\n" + "=" * 60)
    print("ADAPTIVE ASCEND (Filtered Variants)")
    print("=" * 60)
    
    adaptive_ladder = LADDERAdaptive(
        model=model, tokenizer=tokenizer, verifier=verifier,
        variant_generator=variant_generator, grpo_trainer=grpo_trainer,
        adaptive_controller=adaptive_controller,
        max_depth=3, num_variants=3, device=device
    )
    
    adaptive_stats = {
        'total_variants': 0,
        'compute_time': 0.0,
        'solved': 0,
        'total_problems': len(problems),
        'variants_generated': 0,
        'variants_accepted': 0,
        'variants_rejected': 0,
        'compute_saved': 0.0
    }
    
    print("Running adaptive ASCEND...")
    for i, problem in enumerate(problems):
        print(f"  Problem {i+1}/{len(problems)}...")
        start_time = time.time()
        variant_tree, solutions, verification_results = adaptive_ladder.process_problem(problem)
        elapsed = time.time() - start_time
        
        controller_stats = adaptive_controller.get_stats()
        
        adaptive_stats['total_variants'] += len(variant_tree)
        adaptive_stats['compute_time'] += elapsed
        adaptive_stats['solved'] += sum(1 for v in verification_results.values() if v)
        adaptive_stats['variants_generated'] += controller_stats['variants_generated']
        adaptive_stats['variants_accepted'] += controller_stats['variants_accepted']
        adaptive_stats['variants_rejected'] += controller_stats['variants_rejected']
        adaptive_stats['compute_saved'] += controller_stats['compute_saved']
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Variant Statistics:")
    print(f"  Baseline LADDER:")
    print(f"    Total variants: {baseline_stats['total_variants']}")
    print(f"    Variants per problem: {baseline_stats['total_variants'] / baseline_stats['total_problems']:.1f}")
    
    print(f"\n  Adaptive ASCEND:")
    print(f"    Variants generated: {adaptive_stats['variants_generated']}")
    print(f"    Variants accepted: {adaptive_stats['variants_accepted']}")
    print(f"    Variants rejected: {adaptive_stats['variants_rejected']}")
    print(f"    Rejection rate: {adaptive_stats['variants_rejected'] / adaptive_stats['variants_generated'] * 100:.1f}%")
    print(f"    Total variants in tree: {adaptive_stats['total_variants']}")
    print(f"    Variants per problem: {adaptive_stats['total_variants'] / adaptive_stats['total_problems']:.1f}")
    
    variant_reduction = (1 - adaptive_stats['total_variants'] / baseline_stats['total_variants']) * 100
    print(f"\n  Efficiency:")
    print(f"    Variant reduction: {variant_reduction:.1f}%")
    print(f"    Efficiency gain: {baseline_stats['total_variants'] / adaptive_stats['total_variants']:.2f}x")
    
    print(f"\n⏱️  Compute Statistics:")
    print(f"  Baseline LADDER:")
    print(f"    Total time: {baseline_stats['compute_time']:.2f}s")
    print(f"    Time per problem: {baseline_stats['compute_time'] / baseline_stats['total_problems']:.2f}s")
    
    print(f"\n  Adaptive ASCEND:")
    print(f"    Total time: {adaptive_stats['compute_time']:.2f}s")
    print(f"    Time per problem: {adaptive_stats['compute_time'] / adaptive_stats['total_problems']:.2f}s")
    print(f"    Compute saved (estimated): {adaptive_stats['compute_saved']:.2f}s")
    
    time_reduction = (1 - adaptive_stats['compute_time'] / baseline_stats['compute_time']) * 100
    speedup = baseline_stats['compute_time'] / adaptive_stats['compute_time']
    print(f"\n  Efficiency:")
    print(f"    Time reduction: {time_reduction:.1f}%")
    print(f"    Speedup: {speedup:.2f}x")
    
    print(f"\n✅ Accuracy Statistics:")
    baseline_accuracy = baseline_stats['solved'] / baseline_stats['total_variants'] * 100
    adaptive_accuracy = adaptive_stats['solved'] / adaptive_stats['total_variants'] * 100 if adaptive_stats['total_variants'] > 0 else 0
    
    print(f"  Baseline LADDER:")
    print(f"    Solved variants: {baseline_stats['solved']}/{baseline_stats['total_variants']}")
    print(f"    Accuracy: {baseline_accuracy:.1f}%")
    
    print(f"\n  Adaptive ASCEND:")
    print(f"    Solved variants: {adaptive_stats['solved']}/{adaptive_stats['total_variants']}")
    print(f"    Accuracy: {adaptive_accuracy:.1f}%")
    
    print(f"\n🎯 Summary:")
    print(f"  ✓ Variants reduced by {variant_reduction:.1f}%")
    print(f"  ✓ Compute reduced by {time_reduction:.1f}%")
    print(f"  ✓ Speedup: {speedup:.2f}x")
    if abs(adaptive_accuracy - baseline_accuracy) < 5:
        print(f"  ✓ Accuracy maintained ({adaptive_accuracy:.1f}% vs {baseline_accuracy:.1f}%)")
    else:
        print(f"  ⚠ Accuracy changed ({adaptive_accuracy:.1f}% vs {baseline_accuracy:.1f}%)")

if __name__ == '__main__':
    main()
