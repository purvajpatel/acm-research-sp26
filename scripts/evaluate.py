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
from ttrl.ttrl import TTRL
from data.dataset_loader import IntegrationDataset

def evaluate_model(model, tokenizer, problems, verifier, use_ttrl=False, device='cuda'):
    correct = 0
    total = len(problems)
    
    if use_ttrl:
        variant_generator = VariantGenerator(model, tokenizer, device=device)
        ttrl = TTRL(model=model, tokenizer=tokenizer, verifier=verifier,
                   variant_generator=variant_generator, device=device)
    
    for i, problem in enumerate(problems):
        print(f"Evaluating {i+1}/{total}: {problem[:50]}...", end=' ')
        
        if use_ttrl:
            solution, is_verified, _ = ttrl.solve_with_improvement(problem)
        else:
            prompt = f"""Solve the following integration problem step by step.

Problem: {problem}

Provide your solution in the format:
Solution: [antiderivative] + C

Solution:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            import re
            pattern = r'Solution:\s*(.+?)(?=\n\n|\nSolution:|$)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            solution = match.group(1).strip() if match else response.split('\n')[-1].strip()
            
            is_verified, _ = verifier.verify_solution(problem, solution)
        
        if is_verified:
            correct += 1
            print("✓")
        else:
            print("✗")
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate LADDER model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['llama3.2-3b', 'qwen2.5-7b', 'deepseek-r1-7b'])
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['undergraduate', 'mit_bee'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--use_ttrl', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--hf_token', type=str, default=None)
    
    args = parser.parse_args()
    
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
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loading {args.dataset} dataset...")
    dataset_loader = IntegrationDataset()
    if args.dataset == 'undergraduate':
        problems = dataset_loader.get_undergraduate_problems()
    else:
        problems = dataset_loader.get_mit_bee_problems()
    
    print(f"Loaded {len(problems)} problems")
    
    verifier = SolutionVerifier(tolerance=1e-6, num_test_points=100)
    
    print(f"\nEvaluating model{' with TTRL' if args.use_ttrl else ''}...")
    print("=" * 60)
    
    accuracy = evaluate_model(model=model, tokenizer=tokenizer, problems=problems,
                             verifier=verifier, use_ttrl=args.use_ttrl, device=device)
    
    print("\n" + "=" * 60)
    print(f"Final Accuracy: {accuracy:.2%}")
    print("=" * 60)
    
    model_info = ModelLoader.get_model_info(args.model)
    if model_info:
        target = model_info.get('target_accuracy', 0.0)
        baseline = model_info.get('baseline_accuracy', 0.0)
        print(f"\nBaseline (from paper): {baseline:.2%}")
        print(f"Target (from paper): {target:.2%}")
        print(f"Your result: {accuracy:.2%}")
        
        if accuracy >= target:
            print("✓ Target achieved!")
        else:
            print(f"  Still need: {(target - accuracy)*100:.2f}% improvement")

if __name__ == '__main__':
    main()
