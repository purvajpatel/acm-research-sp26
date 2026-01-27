# ASCEND: Adaptive Self-Curriculum Learning

Implementation of ASCEND framework for efficient test-time reinforcement learning.

## Quick Start (Google Colab)

### Step 1: Upload to Google Drive
Upload the entire ASCEND folder to your Google Drive.

### Step 2: Open Colab & Enable GPU
1. Go to https://colab.research.google.com/
2. Create new notebook
3. Enable GPU: **Runtime → Change runtime type → GPU (T4)**

### Step 3: Run This Code

```python
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
import os
os.chdir('/content/drive/MyDrive/ASCEND')  # Adjust path if needed

# Install dependencies
!pip install -r requirements.txt

# Run everything (uses qwen2.5-7b - no authentication needed!)
!python colab_ascend.py
```

**That's it!** The script will:
1. ✅ Collect variant data (20 problems)
2. ✅ Train difficulty estimator (50 epochs)
3. ✅ Compare baseline vs adaptive ASCEND
4. ✅ Show efficiency improvements (2-3x)

### Using Llama Model (Optional)

If you want to use Llama instead:
1. Get HuggingFace token: https://huggingface.co/settings/tokens
2. Request access: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Set token: `os.environ['HUGGINGFACE_TOKEN'] = 'your_token'`
4. Edit `colab_ascend.py`: Change `MODEL_NAME = 'qwen2.5-7b'` to `'llama3.2-3b'`

## Models

- `llama3.2-3b`: Meta Llama 3.2 3B (requires HuggingFace token)
- `qwen2.5-7b`: Qwen 2.5 7B
- `deepseek-r1-7b`: DeepSeek-R1 Distilled 7B

## Datasets

- `undergraduate`: Undergraduate-level integration problems
- `mit_bee`: MIT Integration Bee 2025 problems
