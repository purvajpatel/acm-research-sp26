import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import Tuple, Optional

class ModelLoader:
    MODELS = {
        'llama3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
        'deepseek-r1-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    }
    
    @staticmethod
    def load_model(
        model_name: str,
        device: str = 'auto',
        torch_dtype: torch.dtype = None,
        use_auth_token: Optional[str] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if model_name not in ModelLoader.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(ModelLoader.MODELS.keys())}")
        
        model_path = ModelLoader.MODELS[model_name]
        
        if use_auth_token is None:
            use_auth_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if 'llama' in model_name.lower() and use_auth_token is None:
            raise ValueError(
                "Llama models require HuggingFace authentication token.\n"
                "Set HUGGINGFACE_TOKEN environment variable or pass use_auth_token parameter."
            )
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if torch_dtype is None:
            torch_dtype = torch.float16 if device == 'cuda' else torch.float32
        
        print(f"Loading {model_name} from {model_path}...")
        print(f"Using device: {device}, dtype: {torch_dtype}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=use_auth_token,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if device == 'cuda' and torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map='auto',
                token=use_auth_token,
                trust_remote_code=True
            )
            model._is_distributed = True
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                token=use_auth_token,
                trust_remote_code=True
            )
            model = model.to('cpu')
            model._is_distributed = False
        
        print(f"✓ Loaded {model_name} successfully on {device}")
        return model, tokenizer
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        configs = {
            'llama3.2-3b': {
                'name': 'Llama 3.2 3B',
                'params': '3B',
                'target_dataset': 'undergraduate',
                'baseline_accuracy': 0.01,
                'target_accuracy': 0.82
            },
            'qwen2.5-7b': {
                'name': 'Qwen2.5 7B',
                'params': '7B',
                'target_dataset': 'mit_bee',
                'baseline_accuracy': 0.15,
                'target_accuracy': 0.73
            },
            'deepseek-r1-7b': {
                'name': 'DeepSeek-R1 Distilled Qwen 7B',
                'params': '7B',
                'target_dataset': 'mit_bee',
                'baseline_accuracy': 0.15,
                'target_accuracy': 0.90
            }
        }
        return configs.get(model_name, {})
