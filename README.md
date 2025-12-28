![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Spring 2026 Paper Implementations

**Paper:** [Engineering digital biomarkers of interstitial glucose from noninvasive smartwatches](https://doi.org/10.1038/s41746-021-00465-w)  

**Original Code:** [BIG IDEAs Lab - Glucose Prediction](https://github.com/Big-Ideas-Lab/glucose-prediction)

This paper used consumer wearables (Empatica E4) for glucose prediction without invasive CGM. It employed hybrid feature engineering combining statistical and physiological features from multiple sensors (PPG, EDA, temperature, accelerometer) and food logs, introducing personalized glucose excursion definitions. The work achieved 85.93% accuracy using XGBoost regression.

---

## Setup Instructions

### Clone Original Repository & Install Dependencies
```bash
git clone https://github.com/Big-Ideas-Lab/glucose-prediction.git
```
Follow the installation instructions in the [original repository README](https://github.com/Big-Ideas-Lab/glucose-prediction) for environment setup and dependencies.

### Download Dataset
```bash
aws s3 sync --no-sign-request s3://physionet-open/big-ideas-glycemic-wearable/1.1.2/ ./data/
```
### Add Implementation File
Copy `hypoglycemia_detection.py` into the `glucose-prediction/` directory.

### Run Feature Engineering & Implementation
```bash
python -m src.glucose_fe.cli --config configs/fe_config.yaml --max-workers 1

python hypoglycemia_detection.py
```

---

## Motivation

**Clinical Problem:**
- Prediabetes affects 1 in 3 Americans with 10% annual conversion to Type 2 diabetes, where patients with hypoglycemia (dangerously low blood sugar) can cause confusion, seizures, loss of consciousness, and coma.
- Current solutions are inadequate:
  - Continuous Glucose Monitors (CGMs) cost $6,000+/year and require invasive sensors
  - Only 10% of people with prediabetes know they have the condition
  - No accessible, noninvasive glucose monitoring exists for early intervention

Consumer wearables could enable affordable, continuous glucose monitoring if the minimum sensor requirements for accurate detection can be identified.

---

### My Implementation
This implementation validates the paper's feature engineering approach for a different clinical task: binary classification for hypoglycemia detection (glucose < 70 mg/dL) rather than continuous glucose prediction. The key contribution is a systematic sensor ablation study that tests four configurations: heart rate only, heart rate + temperature, heart rate + EDA, and all sensors combined (HR + temperature + EDA + accelerometer). Each configuration uses the same Random Forest classifier with balanced class weights and 80/20 stratified train-test split. The goal is to understand the tradeoffs between sensor complexity (number of features) and detection performance for hypoglycemia events.


### Limitations
The implementation inherits the small sample size limitations from the original paper (16 participants, 8-10 days each). Additionally, there is class imbalance with only 0.6% hypoglycemia samples (217/38,115), which is realistic but affects metrics. While accuracy is high (98.9%), precision is low (0.211), meaning many false alarms occur. 