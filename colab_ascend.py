"""
ASCEND: Complete Training and Comparison Script for Google Colab
This script does everything: collects data, trains estimator, and shows improvements
"""
import torch
import os
import sys
import json
import time
from pathlib import Path

# Setup paths
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

# Import all necessary modules
from models.model_loader import ModelLoader
from core.verifier import SolutionVerifier
from core.variant_generator import VariantGenerator
from core.grpo_trainer import GRPOTrainer
from core.ladder import LADDER
from core.ladder_adaptive import LADDERAdaptive
from data.dataset_loader import IntegrationDataset
from controllers.adaptive_controller import AdaptiveController
from estimators.difficulty_estimator import DifficultyEstimator
from utils.variant_logger import VariantLogger
import torch.nn as nn

# Import config (if exists)
try:
    from config import DEFAULT_MODEL, DEFAULT_DATASET, MAX_PROBLEMS_COLLECT, MAX_PROBLEMS_COMPARE, ESTIMATOR_EPOCHS
except ImportError:
    # Default values if config doesn't exist
    DEFAULT_MODEL = 'qwen2.5-7b'
    DEFAULT_DATASET = 'undergraduate'
    MAX_PROBLEMS_COLLECT = 20
    MAX_PROBLEMS_COMPARE = 5
    ESTIMATOR_EPOCHS = 50

def compute_similarity(text1: str, text2: str) -> float:
    """Simple similarity metric"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0

def collect_variant_data(model, tokenizer, problems, device='cuda', max_problems=20):
    """Step 1: Collect variant data"""
    print("=" * 60)
    print("STEP 1: Collecting Variant Data")
    print("=" * 60)
    
    # Initialize logger
    logger = VariantLogger("data/variant_logs.json")
    
    # Initialize components
    verifier = SolutionVerifier(tolerance=1e-6, num_test_points=50)
    variant_generator = VariantGenerator(model, tokenizer, device=device)
    grpo_trainer = GRPOTrainer(model=model, tokenizer=tokenizer, learning_rate=1e-5, device=device)
    
    ladder = LADDER(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        variant_generator=variant_generator,
        grpo_trainer=grpo_trainer,
        max_depth=3,
        num_variants=2,
        device=device
    )
    
    # Process problems
    for i, problem in enumerate(problems[:max_problems]):
        print(f"\nProcessing problem {i+1}/{min(len(problems), max_problems)}: {problem[:50]}...")
        
        variant_tree, solutions, verification_results = ladder.process_problem(problem)
        
        # Log all variants
        for node in variant_tree.nodes:
            variant_text = node.problem
            depth = node.depth
            solve_success = verification_results.get(node.id, False)
            
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
        
        if (i + 1) % 5 == 0:
            logger.save()
            stats = logger.get_stats()
            print(f"  Logged {stats['total_variants']} variants so far")
    
    logger.save()
    stats = logger.get_stats()
    print(f"\n✓ Data collection complete!")
    print(f"  Total variants logged: {stats['total_variants']}")
    print(f"  Solve rate: {stats['solve_rate']:.2%}")
    return logger

def train_estimator(logger, epochs=50):
    """Step 2: Train the difficulty estimator"""
    print("\n" + "=" * 60)
    print("STEP 2: Training Difficulty Estimator")
    print("=" * 60)
    
    logs = logger.get_training_data()
    
    if len(logs) < 100:
        print(f"⚠ Warning: Only {len(logs)} examples. Need at least 100.")
        return None
    
    print(f"Loaded {len(logs)} variant examples")
    
    # Prepare data
    X = []
    y_solve = []
    y_cost = []
    y_value = []
    
    for log in logs:
        features = torch.tensor([
            log['tree_depth'],
            log['similarity_to_root'],
            log['variant_length'] / 100.0
        ])
        
        solve_label = 1.0 if log['solve_success'] else 0.0
        cost_label = log['verification_time']
        value_label = log['reward']
        
        X.append(features)
        y_solve.append(solve_label)
        y_cost.append(cost_label)
        y_value.append(value_label)
    
    X = torch.stack(X)
    y_solve = torch.tensor(y_solve)
    y_cost = torch.tensor(y_cost)
    y_value = torch.tensor(y_value)
    
    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_solve_train, y_solve_val = y_solve[:split], y_solve[split:]
    y_cost_train, y_cost_val = y_cost[:split], y_cost[split:]
    y_value_train, y_value_val = y_value[:split], y_value[split:]
    
    # Initialize model
    estimator = DifficultyEstimator()
    optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
    
    solve_loss_fn = nn.BCELoss()
    cost_loss_fn = nn.MSELoss()
    value_loss_fn = nn.BCELoss()
    
    batch_size = 32
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        estimator.train()
        total_loss = 0.0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_solve = y_solve_train[i:i+batch_size]
            batch_cost = y_cost_train[i:i+batch_size]
            batch_value = y_value_train[i:i+batch_size]
            
            variant_embedding = torch.zeros(len(batch_X), estimator.embedding_dim)
            solve_pred, cost_pred, value_pred = estimator(variant_embedding, batch_X)
            
            solve_loss = solve_loss_fn(solve_pred.squeeze(), batch_solve)
            cost_loss = cost_loss_fn(cost_pred.squeeze(), batch_cost)
            value_loss = value_loss_fn(value_pred.squeeze(), batch_value)
            
            total_loss_batch = solve_loss + cost_loss + value_loss
            
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        if (epoch + 1) % 10 == 0:
            estimator.eval()
            with torch.no_grad():
                val_embedding = torch.zeros(len(X_val), estimator.embedding_dim)
                solve_pred, cost_pred, value_pred = estimator(val_embedding, X_val)
                
                val_solve_loss = solve_loss_fn(solve_pred.squeeze(), y_solve_val).item()
                val_cost_loss = cost_loss_fn(cost_pred.squeeze(), y_cost_val).item()
                val_value_loss = value_loss_fn(value_pred.squeeze(), y_value_val).item()
                
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {total_loss/len(X_train)*batch_size:.4f}, "
                      f"Val Loss: {val_solve_loss + val_cost_loss + val_value_loss:.4f}")
    
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(estimator.state_dict(), "checkpoints/estimator.pt")
    print(f"\n✓ Estimator trained and saved to: checkpoints/estimator.pt")
    return estimator

def compare_baseline_adaptive(model, tokenizer, problems, estimator, device='cuda', max_problems=5):
    """Step 3: Compare baseline vs adaptive"""
    print("\n" + "=" * 60)
    print("STEP 3: Comparing Baseline vs Adaptive ASCEND")
    print("=" * 60)
    
    verifier = SolutionVerifier()
    variant_generator = VariantGenerator(model, tokenizer, device=device)
    grpo_trainer = GRPOTrainer(model=model, tokenizer=tokenizer, device=device)
    
    # Initialize adaptive controller
    adaptive_controller = AdaptiveController(estimator)
    
    # Baseline LADDER
    print("\nRunning Baseline LADDER...")
    baseline_ladder = LADDER(
        model=model, tokenizer=tokenizer, verifier=verifier,
        variant_generator=variant_generator, grpo_trainer=grpo_trainer,
        max_depth=3, num_variants=3, device=device
    )
    
    baseline_stats = {
        'total_variants': 0,
        'compute_time': 0.0,
        'solved': 0,
        'total_problems': min(len(problems), max_problems)
    }
    
    for i, problem in enumerate(problems[:max_problems]):
        print(f"  Problem {i+1}/{baseline_stats['total_problems']}...")
        start_time = time.time()
        variant_tree, solutions, verification_results = baseline_ladder.process_problem(problem)
        elapsed = time.time() - start_time
        
        baseline_stats['total_variants'] += len(variant_tree)
        baseline_stats['compute_time'] += elapsed
        baseline_stats['solved'] += sum(1 for v in verification_results.values() if v)
    
    # Adaptive ASCEND
    print("\nRunning Adaptive ASCEND...")
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
        'variants_generated': 0,
        'variants_accepted': 0,
        'variants_rejected': 0,
        'compute_saved': 0.0
    }
    
    for i, problem in enumerate(problems[:max_problems]):
        print(f"  Problem {i+1}/{baseline_stats['total_problems']}...")
        adaptive_controller.reset_stats()
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
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Variant Statistics:")
    print(f"  Baseline LADDER:")
    print(f"    Total variants: {baseline_stats['total_variants']}")
    print(f"    Variants per problem: {baseline_stats['total_variants'] / baseline_stats['total_problems']:.1f}")
    
    print(f"\n  Adaptive ASCEND:")
    print(f"    Variants generated: {adaptive_stats['variants_generated']}")
    print(f"    Variants accepted: {adaptive_stats['variants_accepted']}")
    print(f"    Variants rejected: {adaptive_stats['variants_rejected']}")
    if adaptive_stats['variants_generated'] > 0:
        print(f"    Rejection rate: {adaptive_stats['variants_rejected'] / adaptive_stats['variants_generated'] * 100:.1f}%")
    print(f"    Total variants in tree: {adaptive_stats['total_variants']}")
    print(f"    Variants per problem: {adaptive_stats['total_variants'] / baseline_stats['total_problems']:.1f}")
    
    if baseline_stats['total_variants'] > 0:
        variant_reduction = (1 - adaptive_stats['total_variants'] / baseline_stats['total_variants']) * 100
        efficiency_gain = baseline_stats['total_variants'] / adaptive_stats['total_variants'] if adaptive_stats['total_variants'] > 0 else 0
        print(f"\n  Efficiency:")
        print(f"    Variant reduction: {variant_reduction:.1f}%")
        print(f"    Efficiency gain: {efficiency_gain:.2f}x")
    
    print(f"\n⏱️  Compute Statistics:")
    print(f"  Baseline LADDER:")
    print(f"    Total time: {baseline_stats['compute_time']:.2f}s")
    print(f"    Time per problem: {baseline_stats['compute_time'] / baseline_stats['total_problems']:.2f}s")
    
    print(f"\n  Adaptive ASCEND:")
    print(f"    Total time: {adaptive_stats['compute_time']:.2f}s")
    print(f"    Time per problem: {adaptive_stats['compute_time'] / baseline_stats['total_problems']:.2f}s")
    print(f"    Compute saved (estimated): {adaptive_stats['compute_saved']:.2f}s")
    
    if baseline_stats['compute_time'] > 0:
        time_reduction = (1 - adaptive_stats['compute_time'] / baseline_stats['compute_time']) * 100
        speedup = baseline_stats['compute_time'] / adaptive_stats['compute_time'] if adaptive_stats['compute_time'] > 0 else 0
        print(f"\n  Efficiency:")
        print(f"    Time reduction: {time_reduction:.1f}%")
        print(f"    Speedup: {speedup:.2f}x")
    
    print(f"\n✅ Accuracy Statistics:")
    baseline_accuracy = baseline_stats['solved'] / baseline_stats['total_variants'] * 100 if baseline_stats['total_variants'] > 0 else 0
    adaptive_accuracy = adaptive_stats['solved'] / adaptive_stats['total_variants'] * 100 if adaptive_stats['total_variants'] > 0 else 0
    
    print(f"  Baseline LADDER:")
    print(f"    Solved variants: {baseline_stats['solved']}/{baseline_stats['total_variants']}")
    print(f"    Accuracy: {baseline_accuracy:.1f}%")
    
    print(f"\n  Adaptive ASCEND:")
    print(f"    Solved variants: {adaptive_stats['solved']}/{adaptive_stats['total_variants']}")
    print(f"    Accuracy: {adaptive_accuracy:.1f}%")
    
    print(f"\n🎯 Summary:")
    if baseline_stats['total_variants'] > 0:
        variant_reduction = (1 - adaptive_stats['total_variants'] / baseline_stats['total_variants']) * 100
        print(f"  ✓ Variants reduced by {variant_reduction:.1f}%")
    if baseline_stats['compute_time'] > 0:
        time_reduction = (1 - adaptive_stats['compute_time'] / baseline_stats['compute_time']) * 100
        speedup = baseline_stats['compute_time'] / adaptive_stats['compute_time'] if adaptive_stats['compute_time'] > 0 else 0
        print(f"  ✓ Compute reduced by {time_reduction:.1f}%")
        print(f"  ✓ Speedup: {speedup:.2f}x")
    if abs(adaptive_accuracy - baseline_accuracy) < 5:
        print(f"  ✓ Accuracy maintained ({adaptive_accuracy:.1f}% vs {baseline_accuracy:.1f}%)")
    else:
        print(f"  ⚠ Accuracy changed ({adaptive_accuracy:.1f}% vs {baseline_accuracy:.1f}%)")

def main():
    """Main function - runs everything"""
    print("=" * 60)
    print("ASCEND: Complete Training and Comparison")
    print("=" * 60)
    
    # Configuration (loads from config.py if available)
    MODEL_NAME = DEFAULT_MODEL
    DATASET = DEFAULT_DATASET
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get HuggingFace token (from config.py, environment, or Colab secrets)
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    # Try Colab secrets (best practice for Colab)
    try:
        from google.colab import userdata
        if not hf_token:
            hf_token = userdata.get('HUGGINGFACE_TOKEN')
    except:
        pass  # Not in Colab or secrets not set
    
    if MODEL_NAME == 'llama3.2-3b' and not hf_token:
        print("\n⚠ WARNING: Llama model requires HuggingFace token!")
        print("Please either:")
        print("  1. Set environment variable: export HUGGINGFACE_TOKEN='your_token'")
        print("  2. Or use qwen2.5-7b model (no token needed)")
        print("\nGetting HuggingFace token:")
        print("  1. Go to https://huggingface.co/settings/tokens")
        print("  2. Create a new token (read access)")
        print("  3. Request access to Llama: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        print("  4. Set token in Colab: os.environ['HUGGINGFACE_TOKEN'] = 'your_token'")
        return
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Dataset: {DATASET}")
    print(f"  Device: {DEVICE}")
    print(f"  Collecting data on {MAX_PROBLEMS_COLLECT} problems")
    print(f"  Comparing on {MAX_PROBLEMS_COMPARE} problems")
    if hf_token:
        print(f"  HuggingFace token: {'*' * 10} (set)")
    
    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    try:
        model, tokenizer = ModelLoader.load_model(
            model_name=MODEL_NAME,
            device=DEVICE,
            use_auth_token=hf_token
        )
    except Exception as e:
        if "gated repo" in str(e) or "401" in str(e) or "access" in str(e).lower():
            print("\n❌ ERROR: Cannot access model. Authentication required!")
            print("\nTo fix this:")
            print("1. Get HuggingFace token: https://huggingface.co/settings/tokens")
            print("2. Request access to Llama: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            print("3. In Colab, run:")
            print("   import os")
            print("   os.environ['HUGGINGFACE_TOKEN'] = 'your_token_here'")
            print("4. Or use qwen2.5-7b model (no token needed)")
            print("\nSwitching to qwen2.5-7b (no authentication needed)...")
            MODEL_NAME = 'qwen2.5-7b'
            model, tokenizer = ModelLoader.load_model(
                model_name=MODEL_NAME,
                device=DEVICE,
                use_auth_token=None
            )
        else:
            raise
    
    # Load dataset
    dataset_loader = IntegrationDataset()
    if DATASET == 'undergraduate':
        problems = dataset_loader.get_undergraduate_problems()
    else:
        problems = dataset_loader.get_mit_bee_problems()
    
    print(f"Loaded {len(problems)} problems")
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Step 1: Collect data
    logger = collect_variant_data(model, tokenizer, problems, DEVICE, MAX_PROBLEMS_COLLECT)
    
    # Step 2: Train estimator
    estimator = train_estimator(logger, ESTIMATOR_EPOCHS)
    
    if estimator is None:
        print("\n⚠ Estimator training failed. Using untrained estimator for demo.")
        estimator = DifficultyEstimator()
    
    # Step 3: Compare
    compare_baseline_adaptive(model, tokenizer, problems, estimator, DEVICE, MAX_PROBLEMS_COMPARE)
    
    print("\n" + "=" * 60)
    print("✓ Complete! All steps finished.")
    print("=" * 60)

if __name__ == '__main__':
    main()
