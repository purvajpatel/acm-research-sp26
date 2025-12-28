# hypoglycemia_detection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
HYPOGLYCEMIA_THRESHOLD = 70  # mg/dL
RANDOM_STATE = 42

# Sensor feature groups based on actual column names
HR_FEATURES = ['HR_Mean', 'HR_Std', 'HR_Min', 'HR_Max', 'HR_Q1G', 'HR_Q3G', 'HR_Skew',
               'maxHrv', 'minHrv', 'meanHrv', 'medianHrv', 'sdnn', 'nnx', 'pnnx', 'rmssd']

TEMP_FEATURES = ['TEMP_Mean', 'TEMP_Std', 'TEMP_Min', 'TEMP_Max', 'TEMP_Q1G', 'TEMP_Q3G', 'TEMP_Skew']

EDA_FEATURES = ['EDA_Mean', 'EDA_Std', 'EDA_Min', 'EDA_Max', 'EDA_Q1G', 'EDA_Q3G', 'EDA_Skew',
                'PeakEDA', 'PeakEDA2hr_sum', 'PeakEDA2hr_mean']

ACC_FEATURES = ['ACC_Mean', 'ACC_Std', 'ACC_Min', 'ACC_Max', 'ACC_Q1G', 'ACC_Q3G', 'ACC_Skew',
                'Activity_bouts', 'Activity24', 'Activity1hr', 'ACC_Mean_2hrs', 'ACC_Max_2hrs']

# Demographics (always include)
DEMO_FEATURES = ['Gender', 'HbA1c']

SENSOR_CONFIGS = {
    'config1_hr_only': HR_FEATURES,
    'config2_hr_temp': HR_FEATURES + TEMP_FEATURES,
    'config3_hr_eda': HR_FEATURES + EDA_FEATURES,
    'config4_all_sensors': HR_FEATURES + TEMP_FEATURES + EDA_FEATURES + ACC_FEATURES
}

def load_features(parquet_path):
    """Load the pre-extracted features"""
    print(f"Loading features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df

def create_hypoglycemia_labels(df, glucose_col='Glucose', threshold=70):
    """
    Create binary labels for hypoglycemia detection
    
    Args:
        df: DataFrame with glucose values
        glucose_col: Name of glucose column
        threshold: Hypoglycemia threshold (default 70 mg/dL)
    
    Returns:
        Binary labels (1 = hypoglycemia, 0 = normal)
    """
    labels = (df[glucose_col] < threshold).astype(int)
    print(f"\nClass Distribution:")
    print(f"  Hypoglycemia (<{threshold} mg/dL): {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
    print(f"  Normal (>={threshold} mg/dL): {(labels==0).sum()} ({(labels==0).sum()/len(labels)*100:.1f}%)")
    return labels

def train_evaluate_config(df, y, config_name, feature_cols):
    """
    Train and evaluate model for a given sensor configuration
    """
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"Features: {len(feature_cols)} sensors + {len(DEMO_FEATURES)} demographics")
    print(f"{'='*70}")
    
    # Prepare data - add demographics
    all_features = feature_cols + DEMO_FEATURES
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"Using {len(available_features)} features")
    
    X = df[available_features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': dict(zip(available_features, model.feature_importances_)),
        'n_features': len(available_features)
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    if '1' in metrics['report']:
        print(f"  Precision (Hypoglycemia): {metrics['report']['1']['precision']:.3f}")
        print(f"  Recall (Hypoglycemia): {metrics['report']['1']['recall']:.3f}")
        print(f"  F1-Score (Hypoglycemia): {metrics['report']['1']['f1-score']:.3f}")
    
    return metrics, model

def plot_results(all_results, output_dir='results'):
    """Generate visualizations"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Sensor Configuration Comparison
    configs = list(all_results.keys())
    accuracies = [all_results[c]['accuracy'] for c in configs]
    precisions = [all_results[c]['report'].get('1', {}).get('precision', 0) for c in configs]
    recalls = [all_results[c]['report'].get('1', {}).get('recall', 0) for c in configs]
    f1_scores = [all_results[c]['report'].get('1', {}).get('f1-score', 0) for c in configs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Metrics 
    x = np.arange(len(configs))
    width = 0.2
    
    ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax1.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax1.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Sensor Configuration', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Hypoglycemia Detection: Performance by Sensor Configuration', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('config', 'Config ').replace('_', ' ').title() for c in configs], 
                        rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Number of features vs Accuracy
    n_features = [all_results[c]['n_features'] for c in configs]
    ax2.scatter(n_features, accuracies, s=200, alpha=0.6, c=range(len(configs)), cmap='viridis')
    for i, config in enumerate(configs):
        ax2.annotate(config.replace('config', '').replace('_', ' ').strip(), 
                    (n_features[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Model Complexity', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sensor_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/sensor_comparison.png")
    plt.close()
    
    # 2. Feature Importance for Best Config
    best_config = max(all_results.items(), key=lambda x: x[1]['accuracy'])[0]
    importances = all_results[best_config]['feature_importance']
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    features, scores = zip(*top_features)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(features)))
    ax.barh(range(len(features)), scores, color=colors, alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top 15 Most Important Features\n({best_config.replace("_", " ").title()})', 
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/feature_importance.png")
    plt.close()

def save_results_summary(all_results, output_dir='results'):
    """Save text summary of results"""
    Path(output_dir).mkdir(exist_ok=True)
    
    with open(f'{output_dir}/results_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("HYPOGLYCEMIA DETECTION: SENSOR ABLATION STUDY RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Threshold: <{HYPOGLYCEMIA_THRESHOLD} mg/dL\n")
        f.write(f"Model: Random Forest (100 trees, max_depth=10, class_weight='balanced')\n")
        f.write(f"Dataset: BIG IDEAs Lab Glycemic Variability\n")
        f.write(f"Validation: 80/20 train-test split\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS BY CONFIGURATION:\n")
        f.write("-"*70 + "\n\n")
        
        for config, results in all_results.items():
            f.write(f"\n{config.upper()}:\n")
            f.write(f"  Number of Features: {results['n_features']}\n")
            f.write(f"  Accuracy:           {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)\n")
            if '1' in results['report']:
                f.write(f"  Precision:          {results['report']['1']['precision']:.3f}\n")
                f.write(f"  Recall:             {results['report']['1']['recall']:.3f}\n")
                f.write(f"  F1-Score:           {results['report']['1']['f1-score']:.3f}\n")
        
        # Find best configuration
        best_config = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        f.write(f"\n\n" + "="*70 + "\n")
        f.write(f"BEST CONFIGURATION: {best_config[0].upper()}\n")
        f.write(f"="*70 + "\n")
        f.write(f"Accuracy: {best_config[1]['accuracy']:.1%}\n")
        f.write(f"Number of Features: {best_config[1]['n_features']}\n")
        
        # Analysis
        f.write(f"\n\nKEY FINDINGS:\n")
        f.write("-"*70 + "\n")
        
        # Compare HR only vs all sensors
        hr_only_acc = all_results.get('config1_hr_only', {}).get('accuracy', 0)
        all_sensors_acc = all_results.get('config4_all_sensors', {}).get('accuracy', 0)
        improvement = (all_sensors_acc - hr_only_acc) * 100
        
        f.write(f"1. Heart rate alone achieves {hr_only_acc:.1%} accuracy\n")
        f.write(f"2. Adding all sensors improves accuracy by {improvement:.1f} percentage points\n")
        f.write(f"3. Best configuration uses {best_config[1]['n_features']} features\n")
        
        # Top features
        f.write(f"\n\nTOP 5 MOST IMPORTANT FEATURES:\n")
        f.write("-"*70 + "\n")
        importances = best_config[1]['feature_importance']
        top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (feat, imp) in enumerate(top5, 1):
            f.write(f"{i}. {feat}: {imp:.4f}\n")
    
    print(f"✓ Saved: {output_dir}/results_summary.txt")

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("HYPOGLYCEMIA DETECTION WITH SENSOR ABLATION STUDY")
    print("="*70 + "\n")
    
    # Load pre-extracted features
    features_path = "out/ALL_features_cleaned.parquet"
    df = load_features(features_path)
    
    # Create  labels
    y = create_hypoglycemia_labels(df)
    
    # Test each configuration
    all_results = {}
    all_models = {}
    
    for config_name, feature_cols in SENSOR_CONFIGS.items():
        metrics, model = train_evaluate_config(df, y, config_name, feature_cols)
        all_results[config_name] = metrics
        all_models[config_name] = model
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS...")
    print(f"{'='*70}")
    plot_results(all_results)
    
    # Save summary
    save_results_summary(all_results)
    
    print("\n" + "="*70)
    print("✓ COMPLETE! Check the results/ folder for all outputs")
    print("="*70 + "\n")
    
    # Print
    print("QUICK SUMMARY:")
    print("-" * 70)
    for config, results in all_results.items():
        print(f"{config:25s}: {results['accuracy']:.1%} accuracy ({results['n_features']} features)")
    
    best = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✓ Best: {best[0]} with {best[1]['accuracy']:.1%} accuracy")

if __name__ == "__main__":
    main()