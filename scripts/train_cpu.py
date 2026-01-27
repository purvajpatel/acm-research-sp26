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

def main():
    parser = argparse.ArgumentParser(description='Train LADDER model on CPU')
    parser.add_argument('--model', type=str, default='llama3.2-3b',
                       choices=['llama3.2-3b', 'qwen2.5-7b', 'deepseek-r1-7b'])
    parser.add_argument('--dataset', type=str, default='undergraduate',
                       choices=['undergraduate', 'mit_bee'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_depth', type=int, default=2)
    parser.add_argument('--num_variants', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--max_problems', type=int, default=5,
                       help='Max problems to process (for testing)')
    
    args = parser.parse_args()
    
    device = 'cpu'
    print(f"Using device: {device}")
    print("⚠ CPU training is very slow. Consider using Google Colab for GPU.")
    
    print(f"Loading {args.model}...")
    model, tokenizer = ModelLoader.load_model(
        model_name=args.model,
        device=device,
        use_auth_token=args.hf_token or os.getenv('HUGGINGFACE_TOKEN')
    )
    
    print(f"Loading {args.dataset} dataset...")
    dataset_loader = IntegrationDataset()
    if args.dataset == 'undergraduate':
        problems = dataset_loader.get_undergraduate_problems()
    else:
        problems = dataset_loader.get_mit_bee_problems()
    
    problems = problems[:args.max_problems]
    print(f"Processing {len(problems)} problems (limited for CPU)")
    
    print("Initializing LADDER components...")
    verifier = SolutionVerifier(tolerance=1e-6, num_test_points=50)
    variant_generator = VariantGenerator(model, tokenizer, device=device)
    grpo_trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        device=device
    )
    
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        epoch_losses = []
        epoch_stats = {'total_verified': 0, 'total_variants': 0, 'problems_processed': 0}
        
        for i, problem in enumerate(problems):
            print(f"\nProcessing problem {i+1}/{len(problems)}: {problem[:50]}...")
            
            try:
                loss, stats = ladder.train_on_problem(problem)
                
                epoch_losses.append(loss)
                epoch_stats['total_verified'] += stats['num_verified']
                epoch_stats['total_variants'] += stats['num_total']
                epoch_stats['problems_processed'] += 1
                
                print(f"  Loss: {loss:.6f}")
                print(f"  Verified: {stats['num_verified']}/{stats['num_total']}")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Total Verified: {epoch_stats['total_verified']}/{epoch_stats['total_variants']}")
            if epoch_stats['total_variants'] > 0:
                print(f"  Verification Rate: {epoch_stats['total_verified']/epoch_stats['total_variants']*100:.2f}%")
            
            checkpoint_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_epoch_{epoch+1}.pt")
            grpo_trainer.save_checkpoint(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    print(f"\n✓ Training complete!")

if __name__ == '__main__':
    main()
