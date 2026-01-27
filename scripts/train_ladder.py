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
    parser = argparse.ArgumentParser(description='Train LADDER model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['llama3.2-3b', 'qwen2.5-7b', 'deepseek-r1-7b'])
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['undergraduate', 'mit_bee'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--num_variants', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--hf_token', type=str, default=None)
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
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
    
    print(f"Loaded {len(problems)} problems")
    
    print("Initializing LADDER components...")
    verifier = SolutionVerifier(tolerance=1e-6, num_test_points=100)
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
            
            loss, stats = ladder.train_on_problem(problem)
            
            epoch_losses.append(loss)
            epoch_stats['total_verified'] += stats['num_verified']
            epoch_stats['total_variants'] += stats['num_total']
            epoch_stats['problems_processed'] += 1
            
            print(f"  Loss: {loss:.6f}")
            print(f"  Verified: {stats['num_verified']}/{stats['num_total']}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Total Verified: {epoch_stats['total_verified']}/{epoch_stats['total_variants']}")
        print(f"  Verification Rate: {epoch_stats['total_verified']/epoch_stats['total_variants']*100:.2f}%")
        
        checkpoint_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_epoch_{epoch+1}.pt")
        grpo_trainer.save_checkpoint(checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    final_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_final.pt")
    grpo_trainer.save_checkpoint(final_path)
    print(f"\n✓ Training complete! Final model saved to: {final_path}")

if __name__ == '__main__':
    main()
