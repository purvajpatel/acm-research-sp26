"""
RUN ALL - Execute complete pipeline in sequence
This script runs all notebooks (01-08) in order.
"""
import os
import sys
import subprocess
import time

scripts = [
    ('scripts/01_setup.py', 'Setup & Environment Verification'),
    ('scripts/02_data_graph.py', 'Data Loading & Graph Construction'),
    ('scripts/03_train_gat_baseline.py', 'Train Baseline GAT Model'),
    ('scripts/04_eval_baseline.py', 'Evaluate Baseline Model'),
    ('scripts/05_quantum_feature_map.py', 'Quantum Feature Mapping'),
    ('scripts/06_train_gat_quantum.py', 'Train Quantum GAT Model'),
    ('scripts/07_eval_quantum.py', 'Evaluate & Compare Models'),
    ('scripts/08_explain_llm.py', 'Generate Explanations'),
]

def run_script(script_name, description):
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    result = subprocess.run(
        [sys.executable, script_name],
        env=env,
        capture_output=False
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}")
        print(f"   Elapsed time: {elapsed:.2f}s")
        return False
    
    print(f"\n✅ SUCCESS: {script_name} completed in {elapsed:.2f}s")
    return True

def main():
    print("="*70)
    print("KRYPTOS - QUANTUM BLOCKCHAIN FRAUD DETECTION")
    print("Complete Pipeline Execution")
    print("="*70)
    
    total_start = time.time()
    
    # Ask user which scripts to run
    print("\nAvailable scripts:")
    for i, (script, desc) in enumerate(scripts, 1):
        print(f"  {i}. {script:<30} - {desc}")
    
    print("\nOptions:")
    print("  [A] Run all scripts")
    print("  [1-8] Run specific script")
    print("  [Q] Quit")
    
    choice = input("\nYour choice: ").strip().upper()
    
    if choice == 'Q':
        print("Exiting...")
        return
    elif choice == 'A':
        # Run all scripts
        for script, description in scripts:
            success = run_script(script, description)
            if not success:
                print(f"\n❌ Pipeline stopped due to error in {script}")
                return
    elif choice.isdigit() and 1 <= int(choice) <= len(scripts):
        # Run specific script
        idx = int(choice) - 1
        script, description = scripts[idx]
        run_script(script, description)
    else:
        print("Invalid choice!")
        return
    
    total_elapsed = time.time() - total_start
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {minutes}m {seconds}s")
    print("\n📊 Check the following directories:")
    print("  • artifacts/ - Saved models and metrics")
    print("  • figures/   - Generated visualizations")

if __name__ == '__main__':
    main()
