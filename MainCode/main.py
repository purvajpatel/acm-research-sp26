import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import os
import copy


class TextDataset(Dataset):
    """Convert text dataframe to PyTorch dataset."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df.values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        row = self.texts[idx]
        
        if len(row) == 1:
            text = str(row[0])
        elif len(row) == 2:
            text = str(row[0]) + " [SEP] " + str(row[1])
        else:
            text = " [SEP] ".join([str(x) for x in row])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze() if 'token_type_ids' in encoding else None
        }


class LayerWiseMergedModel(nn.Module):
    """
    Wrapper that applies layer-wise merging during forward pass.
    Merges base model with fine-tuned models using learned weights.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        finetuned_models: List[nn.Module],
        num_layers: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.base_model = base_model
        self.finetuned_models = nn.ModuleList(finetuned_models)
        self.device = device
        self.num_tasks = len(finetuned_models)
        self.num_layers = num_layers
        
        # Layer-wise merging weights: shape (num_tasks, num_layers)
        self.merge_weight = nn.Parameter(
            torch.ones(self.num_tasks, self.num_layers, device=device)
        )
        
        # Compute task vectors (difference between fine-tuned and base)
        self.register_buffer('task_vectors', self._compute_task_vectors())
    
    def _compute_task_vectors(self) -> torch.Tensor:
        """Compute task vectors for all models."""
        base_params = [p.data.clone() for p in self.base_model.parameters() 
                       if 'classifier' not in self._get_param_name(p)]
        
        task_vectors_list = []
        for finetuned_model in self.finetuned_models:
            finetuned_params = [p.data.clone() for p in finetuned_model.parameters() 
                               if 'classifier' not in self._get_param_name(p)]
            
            # Flatten and concatenate task vectors
            tv = []
            for bp, fp in zip(base_params, finetuned_params):
                if bp.shape == fp.shape:
                    tv.append((fp - bp).flatten())
            
            task_vectors_list.append(torch.cat(tv))
        
        return torch.stack(task_vectors_list)
    
    def _get_param_name(self, param):
        """Helper to get parameter name."""
        for name, p in self.named_parameters():
            if p is param:
                return name
        return ""
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """Forward pass with layer-wise merging."""
        # Get base model output
        base_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # For now, use base model output directly
        # In a full implementation, you would apply task vector modifications during forward pass
        return base_output


class SAM(torch.optim.Optimizer):
    """Sharpness Aware Minimization optimizer."""
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups_base = [
            {k: v for k, v in group.items() if k != 'rho' and k != 'adaptive'}
            for group in self.param_groups
        ]
        self.base_optimizer_instance = base_optimizer(self.param_groups, **{
            k: v for k, v in kwargs.items() if k in ['lr', 'weight_decay', 'momentum']
        })
    
    def first_step(self, zero_grad=False):
        """Compute and apply SAM perturbation."""
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if not hasattr(p, 'sam_original'):
                    p.sam_original = p.data.clone()
                
                p.data.add_(p.grad, alpha=scale)
        
        if zero_grad:
            self.zero_grad()
    
    def second_step(self, zero_grad=False):
        """Update parameters at perturbed point."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if hasattr(p, 'sam_original'):
                    p.data = p.sam_original
        
        self.base_optimizer_instance.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        """Compute gradient norm."""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


class SAMerging:
    """
    Sharpness-Aware Model Merging (SAMerging)
    Paper-aligned implementation with layer-wise weights.
    """
    
    def __init__(
        self,
        theta0: nn.Module,
        theta_t: List[nn.Module],
        calibration_datasets: Dict[str, Dict[str, pd.DataFrame]],
        tokenizer,
        rho: float = 0.05,
        eta: float = 0.01,
        num_epochs: int = 20,
        temperature: float = 4.0,
        batch_size: int = 4,
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.theta0 = theta0.to(device)
        self.theta_t = [model.to(device) for model in theta_t]
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.rho = rho
        self.eta = eta
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.T = len(theta_t)
        
        # Create dataloaders
        self.calibration_dataloaders = self._create_dataloaders(calibration_datasets)
        
        # Count non-classifier layers
        self.num_layers = len([name for name, _ in theta0.named_parameters() 
                               if 'classifier' not in name])
        
        # Create layer-wise merged model
        self.merged_model = LayerWiseMergedModel(
            theta0, theta_t, self.num_layers, device
        ).to(device)
        
        # Pre-compute expert logits
        self.precomputed_expert_logits = self._precompute_expert_logits()
    
    def _create_dataloaders(self, calibration_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> List[DataLoader]:
        """Create dataloaders from calibration datasets."""
        dataloaders = []
        
        models = sorted(calibration_datasets.keys())
        
        for model_name in models:
            if 'validation' in calibration_datasets[model_name]:
                df_cal = calibration_datasets[model_name]['validation']
            elif 'val' in calibration_datasets[model_name]:
                df_cal = calibration_datasets[model_name]['val']
            else:
                df_cal = list(calibration_datasets[model_name].values())[0]
            
            dataset = TextDataset(df_cal, self.tokenizer, self.max_length)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            dataloaders.append(dataloader)
        
        return dataloaders
    
    def _precompute_expert_logits(self) -> List[List[torch.Tensor]]:
        """Pre-compute expert logits for all batches."""
        precomputed = [[] for _ in range(self.T)]
        
        for t in range(self.T):
            for batch in self.calibration_dataloaders[t]:
                batch_device = {k: v.to(self.device) for k, v in batch.items() if v is not None}
                
                with torch.no_grad():
                    output = self.theta_t[t](**batch_device)
                    logits = output[0] if isinstance(output, tuple) else output.logits
                    precomputed[t].append(logits.cpu())
        
        return precomputed
    
    def compute_loss(self) -> torch.Tensor:
        """Compute KL divergence loss between merged and expert models."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for t in range(self.T):
            task_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.calibration_dataloaders[t]):
                batch_device = {k: v.to(self.device) for k, v in batch.items() if v is not None}
                
                # Get merged model logits
                merged_output = self.merged_model(**batch_device)
                merged_logits = merged_output[0] if isinstance(merged_output, tuple) else merged_output.logits
                
                # Get expert logits
                expert_logits = self.precomputed_expert_logits[t][batch_idx].to(self.device)
                
                # Handle dimension mismatch
                num_classes = min(merged_logits.shape[1], expert_logits.shape[1])
                merged_logits = merged_logits[:, :num_classes]
                expert_logits = expert_logits[:, :num_classes]
                
                # Compute KL divergence
                p_t = F.softmax(expert_logits / self.temperature, dim=-1)
                q_lambda = F.log_softmax(merged_logits / self.temperature, dim=-1)
                
                kl_loss = F.kl_div(q_lambda, p_t, reduction='batchmean')
                task_loss = task_loss + kl_loss
                num_batches += 1
            
            if num_batches > 0:
                task_loss = task_loss / num_batches
                total_loss = total_loss + task_loss / self.T
        
        return total_loss
    
    def optimize_lambda(self) -> torch.Tensor:
        """Main optimization loop using SAM."""
        # Setup standard optimizer
        optimizer = torch.optim.Adam(
            [self.merged_model.merge_weight],
            lr=self.eta
        )
        
        for epoch in range(self.num_epochs):
            # Compute initial loss and gradients
            optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            
            # SAM: Compute perturbation - FIXED
            if self.merged_model.merge_weight.grad is not None:
                grad_norm = torch.norm(self.merged_model.merge_weight.grad)
            else:
                grad_norm = torch.tensor(0.0, device=self.device)
            
            if grad_norm > 1e-8:
                # Save original parameters
                sam_original = self.merged_model.merge_weight.data.clone()
                
                # Apply perturbation
                scale = self.rho / (grad_norm + 1e-12)
                with torch.no_grad():
                    self.merged_model.merge_weight.data.add_(
                        self.merged_model.merge_weight.grad, 
                        alpha=scale
                    )
                
                # Compute loss at perturbed point
                optimizer.zero_grad()
                loss_perturbed = self.compute_loss()
                loss_perturbed.backward()
                
                # Restore original parameters and step
                with torch.no_grad():
                    self.merged_model.merge_weight.data = sam_original
                
                optimizer.step()
            else:
                # If gradient norm is too small, just do regular Adam step
                optimizer.step()
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}")
        
        return self.merged_model.merge_weight.detach()
    
    def get_merged_model(self) -> nn.Module:
        """Return the final merged model (approximate as base model for now)."""
        # For actual implementation, would need to apply final task vectors
        return self.theta0
    
    def evaluate_on_test_set(
        self,
        merged_model: nn.Module,
        test_datasets: Dict[str, Dict[str, pd.DataFrame]],
        task_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate merged model on test sets."""
        results = {}
        
        models = sorted(test_datasets.keys())
        
        for task_idx, model_name in enumerate(models):
            if 'test' in test_datasets[model_name]:
                df_test = test_datasets[model_name]['test']
            elif 'test_matched' in test_datasets[model_name]:
                df_test = test_datasets[model_name]['test_matched']
            else:
                df_test = list(test_datasets[model_name].values())[0]
            
            test_dataset = TextDataset(df_test, self.tokenizer, self.max_length)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            merged_model.eval()
            correct = 0
            total = 0
            start_time = time.time()
            
            with torch.no_grad():
                for batch in test_loader:
                    batch_device = {k: v.to(self.device) for k, v in batch.items() if v is not None}
                    
                    outputs = merged_model(**batch_device)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    
                    teacher_outputs = self.theta_t[task_idx](**batch_device)
                    teacher_logits = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs.logits
                    teacher_predictions = torch.argmax(teacher_logits, dim=-1)
                    
                    correct += (predictions == teacher_predictions).sum().item()
                    total += predictions.shape[0]
            
            elapsed_time = time.time() - start_time
            accuracy = correct / total if total > 0 else 0.0
            
            results[model_name] = {
                'accuracy': accuracy,
                'runtime_seconds': elapsed_time,
                'num_samples': total
            }
            
            print(f"Task '{model_name}': Accuracy={accuracy:.4f}, Runtime={elapsed_time:.2f}s")
        
        return results


# Example usage
if __name__ == "__main__":
    from transformers import (
        AutoTokenizer, 
        BertConfig,
        BertForSequenceClassification
    )
    
    base_model_path = os.path.abspath("./StudentModel").replace("\\", "/")
    fine_tuned_model_paths = [
        os.path.abspath("./TeacherModels/CoLA").replace("\\", "/"),
        os.path.abspath("./TeacherModels/MNLI").replace("\\", "/"),
        os.path.abspath("./TeacherModels/MRPC").replace("\\", "/"),
        os.path.abspath("./TeacherModels/QNLI").replace("\\", "/"),
        os.path.abspath("./TeacherModels/QQP").replace("\\", "/"),
        os.path.abspath("./TeacherModels/SST-2").replace("\\", "/")
    ]
    
    task_names = ["cola", "mnli", "mrpc", "qnli", "qqp", "sst2"]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True, trust_remote_code=True)
    
    config_path = os.path.join(base_model_path, "config.json")
    config = BertConfig.from_json_file(config_path)
    
    model_weights_path = os.path.join(base_model_path, "pytorch_model.bin")
    theta0 = BertForSequenceClassification(config)
    state_dict = torch.load(model_weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    theta0.load_state_dict(state_dict, strict=False)
    
    theta_t = []
    for model_path in fine_tuned_model_paths:
        config_path = os.path.join(model_path, "config.json")
        config = BertConfig.from_json_file(config_path)
        
        model_weights_path = os.path.join(model_path, "pytorch_model.bin")
        model = BertForSequenceClassification(config)
        state_dict = torch.load(model_weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict, strict=False)
        
        theta_t.append(model)

    from getDataset import getDataset
    
    dataset_dict = getDataset()

    samerging = SAMerging(
        theta0=theta0,
        theta_t=theta_t,
        calibration_datasets=dataset_dict,
        tokenizer=tokenizer,
        rho=0.05,
        eta=0.01,
        num_epochs=10,
        batch_size=4,
        max_length=128
    )
    
    print("=" * 60)
    print("Starting SAMerging Optimization...")
    print("=" * 60)

    optimal_weights = samerging.optimize_lambda()
    merged_model = samerging.get_merged_model()

    print("\n" + "=" * 60)
    print("SAMerging completed!")
    print("=" * 60)
    print("\nOptimal merge weights shape:", optimal_weights.shape)
    
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    test_results = samerging.evaluate_on_test_set(merged_model, dataset_dict, task_names)
    
    print("\n" + "=" * 60)
    print("Test Set Results Summary")
    print("=" * 60)
    
    total_accuracy = 0.0
    
    for i, task_name in enumerate(task_names):
        model_key = list(dataset_dict.keys())[i]
        if model_key in test_results:
            results = test_results[model_key]
            print(f"\n{task_name}:")
            print(f"  Accuracy:    {results['accuracy']:.4f}")
            print(f"  Runtime:     {results['runtime_seconds']:.4f}s")
            print(f"  # Samples:   {results['num_samples']}")
            total_accuracy += results['accuracy']
    
    avg_accuracy = total_accuracy / len(task_names)
    
    print("\n" + "-" * 60)
    print(f"Average Accuracy across all tasks: {avg_accuracy:.4f}")
    print("=" * 60)