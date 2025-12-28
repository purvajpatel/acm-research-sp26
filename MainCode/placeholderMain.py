import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
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
    Merges BERT backbones while keeping task-specific heads separate.
    Uses functional_call approach to maintain gradient flow.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        finetuned_models: List[nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.num_tasks = len(finetuned_models)
        
        # Extract backbones and heads
        self.pretrained_backbone = self._extract_backbone(base_model).requires_grad_(False)
        self.task_heads = nn.ModuleList([self._extract_head(model) for model in finetuned_models])
        
        # Compute task vectors (θ_t - θ_0) for each fine-tuned model
        self.task_vectors = nn.ModuleList()
        for ft_model in finetuned_models:
            ft_backbone = self._extract_backbone(ft_model)
            task_vector_model = copy.deepcopy(ft_backbone)
            
            # Compute task vectors: τ = θ_finetuned - θ_pretrained
            for (name, ft_param), (_, base_param) in zip(
                ft_backbone.named_parameters(), 
                self.pretrained_backbone.named_parameters()
            ):
                # Get the corresponding parameter in task_vector_model
                param_parts = name.split('.')
                target_param = task_vector_model
                for part in param_parts[:-1]:
                    target_param = getattr(target_param, part)
                
                # Set the task vector
                setattr(target_param, param_parts[-1], 
                       nn.Parameter((ft_param - base_param).detach().requires_grad_(False)))
            
            self.task_vectors.append(task_vector_model.requires_grad_(False))
        
        # Count layers in backbone
        self.num_layers = self._count_layers(self.pretrained_backbone)
        
        # Layer-wise merge weights: (num_tasks, num_layers)
        init_value = 1.0 / self.num_tasks
        self.merge_weight = nn.Parameter(
            torch.full((self.num_tasks, self.num_layers), init_value, device=device)
        )
    
    def _extract_backbone(self, model: nn.Module) -> nn.Module:
        """Extract the backbone (BERT encoder) from the model."""
        if hasattr(model, 'bert'):
            return model.bert
        elif hasattr(model, 'transformer'):
            return model.transformer
        elif hasattr(model, 'gpt2'):
            return model.gpt2
        else:
            return model
    
    def _extract_head(self, model: nn.Module) -> nn.Module:
        """Extract the classification head from the model."""
        if hasattr(model, 'classifier'):
            return model.classifier
        elif hasattr(model, 'lm_head'):
            return model.lm_head
        else:
            return nn.Identity()
    
    def _count_layers(self, backbone: nn.Module) -> int:
        """Count transformer layers in backbone."""
        if hasattr(backbone, 'encoder'):  # BERT
            return len(backbone.encoder.layer)
        elif hasattr(backbone, 'h'):  # GPT-2
            return len(backbone.h)
        else:
            return 12
    
    def _get_layer_index_for_param(self, param_name: str) -> Optional[int]:
        """Extract layer index from parameter name."""
        parts = param_name.split('.')
        for i, part in enumerate(parts):
            if part == 'layer' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except (ValueError, IndexError):
                    pass
            elif part == 'h' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except (ValueError, IndexError):
                    pass
        return None
    
    def merge_weights(self):
        """
        Merge task vectors with pretrained model to create merged state dict.
        This is called before forward pass to update the merged parameters.
        """
        # Start with pretrained model's state dict
        merged_state_dict = {
            name: param.clone() 
            for name, param in self.pretrained_backbone.named_parameters()
        }
        
        # Add weighted task vectors
        # merge_weight shape: (num_tasks, num_layers)
        for task_idx in range(self.num_tasks):
            task_vector = self.task_vectors[task_idx]
            
            layer_idx = 0
            for name, tv_param in task_vector.named_parameters():
                if name in merged_state_dict:
                    # Get the weight for this task and layer
                    param_layer_idx = self._get_layer_index_for_param(name)
                    if param_layer_idx is not None:
                        weight = self.merge_weight[task_idx, param_layer_idx]
                    else:
                        # For non-layer parameters, use mean weight
                        weight = self.merge_weight[task_idx].mean()
                    
                    # Add weighted task vector
                    merged_state_dict[name] = merged_state_dict[name] + weight * tv_param
                    layer_idx += 1
        
        self._merged_state_dict = merged_state_dict
        return merged_state_dict
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, task_idx: int = 0):
        """
        Forward pass using functional_call with merged parameters.
        """
        from torch.func import functional_call
        
        # Use functional_call to forward with merged parameters
        backbone_output = functional_call(
            self.pretrained_backbone,
            self._merged_state_dict,
            args=(input_ids,),
            kwargs={
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'return_dict': True
            }
        )
        
        hidden_states = backbone_output.last_hidden_state
        task_head = self.task_heads[task_idx]
        logits = task_head(hidden_states[:, 0, :])
        
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(logits=logits)
    
    def get_merged_backbone(self) -> nn.Module:
        """Create final merged backbone with task vectors baked in."""
        merged_backbone = copy.deepcopy(self.pretrained_backbone)
        merged_backbone.load_state_dict(self._merged_state_dict)
        return merged_backbone


class SAMerging:
    """
    Sharpness-Aware Model Merging for multi-task BERT models.
    """
    
    def __init__(
        self,
        theta0: nn.Module,
        theta_t: List[nn.Module],
        calibration_datasets: Dict[str, Dict[str, pd.DataFrame]],
        tokenizer,
        eta: float = 0.01,
        num_epochs: int = 20,
        temperature: float = 1.0,
        batch_size: int = 4,
        max_length: int = 128,
        rho: float = 0.07,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.theta0 = theta0.to(device)
        self.theta_t = [model.to(device) for model in theta_t]
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.eta = eta
        self.num_epochs = num_epochs
        self.temperature = temperature
        self.rho = rho
        self.T = len(theta_t)
        self.calibration_datasets = calibration_datasets
        
        # Create merged model
        self.merged_model = LayerWiseMergedModel(theta0, theta_t, device).to(device)
        
        # Create infinite batch iterators for each task
        self._setup_batch_iterators()
    
    def _setup_batch_iterators(self):
        """Setup infinite batch iterators for each task."""
        self.batch_iterators = {}
        
        models = sorted(self.calibration_datasets.keys())
        
        for model_name in models:
            # Get calibration dataset
            if 'validation' in self.calibration_datasets[model_name]:
                df_cal = self.calibration_datasets[model_name]['validation']
            elif 'val' in self.calibration_datasets[model_name]:
                df_cal = self.calibration_datasets[model_name]['val']
            else:
                df_cal = list(self.calibration_datasets[model_name].values())[0]
            
            dataset = TextDataset(df_cal, self.tokenizer, self.max_length)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Create infinite iterator
            self.batch_iterators[model_name] = self._infinite_dataloader(dataloader)
    
    @staticmethod
    def _infinite_dataloader(dataloader):
        """Create infinite iterator from dataloader."""
        while True:
            for batch in dataloader:
                yield batch
    
    def compute_kl_loss(self) -> torch.Tensor:
        """
        Compute multi-teacher KL divergence loss.
        L_KD(λ) = Σ_t α_t E_{x∈B_t}[KL(p_t(·|x) ∥ q_λ(·|x))]
        """
        total_loss = 0.0
        models = sorted(self.calibration_datasets.keys())
        alpha = 1.0 / self.T
        
        for task_idx, model_name in enumerate(models):
            batch = next(self.batch_iterators[model_name])
            batch_device = {k: v.to(self.device) for k, v in batch.items() if v is not None}
            
            # Forward through merged model
            merged_output = self.merged_model(
                input_ids=batch_device['input_ids'],
                attention_mask=batch_device.get('attention_mask'),
                token_type_ids=batch_device.get('token_type_ids'),
                task_idx=task_idx
            )
            merged_logits = merged_output.logits
            
            # Forward through expert (fine-tuned) model
            with torch.no_grad():
                expert_output = self.theta_t[task_idx](**batch_device)
                expert_logits = expert_output.logits
            
            # Match batch sizes and number of classes
            batch_size = min(merged_logits.shape[0], expert_logits.shape[0])
            merged_logits = merged_logits[:batch_size]
            expert_logits = expert_logits[:batch_size]
            
            num_classes = min(merged_logits.shape[1], expert_logits.shape[1])
            merged_logits = merged_logits[:, :num_classes]
            expert_logits = expert_logits[:, :num_classes]
            
            # Compute KL(p_t ∥ q_λ) where p_t is teacher and q_λ is student
            p_teacher = F.softmax(expert_logits / self.temperature, dim=-1)
            q_student = F.log_softmax(merged_logits / self.temperature, dim=-1)
            
            kl_loss = F.kl_div(q_student, p_teacher, reduction='batchmean')
            total_loss = total_loss + alpha * kl_loss
        
        return total_loss
    
    def optimize_weights(self) -> torch.Tensor:
        """
        Optimize merge weights using Sharpness-Aware Minimization (SAM).
        """
        # Only optimize merge_weight parameter
        params_to_optimize = [self.merged_model.merge_weight]
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.eta)
        
        for epoch in range(self.num_epochs):
            # Merge weights before computing loss
            self.merged_model.merge_weights()
            
            # ===== SAM Ascent Step =====
            optimizer.zero_grad()
            loss = self.compute_kl_loss()
            loss.backward()
            
            # Check if we have gradients
            if self.merged_model.merge_weight.grad is None:
                print(f"Warning: No gradient at epoch {epoch + 1}")
                continue
            
            grad = self.merged_model.merge_weight.grad.clone()
            grad_norm = torch.norm(grad)
            
            if grad_norm > 1e-12:
                # Compute perturbation: ε = ρ * g / ||g||_2
                epsilon = (self.rho / grad_norm) * grad
                
                # Apply perturbation to weights
                with torch.no_grad():
                    self.merged_model.merge_weight.data.add_(epsilon)
                
                # Merge weights with perturbed weights
                self.merged_model.merge_weights()
                
                # ===== SAM Descent Step =====
                optimizer.zero_grad()
                loss_perturbed = self.compute_kl_loss()
                loss_perturbed.backward()
                
                # Remove perturbation
                with torch.no_grad():
                    self.merged_model.merge_weight.data.sub_(epsilon)
                
                # Update weights using gradient from perturbed point
                optimizer.step()
            else:
                # If gradient is too small, just do regular update
                optimizer.step()
            
            # Log progress
            grad_norm_current = self.merged_model.merge_weight.grad.norm().item() if self.merged_model.merge_weight.grad is not None else 0.0
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}, "
                  f"Gradient norm: {grad_norm_current:.6f}, "
                  f"Weights - Min: {self.merged_model.merge_weight.min():.4f}, "
                  f"Max: {self.merged_model.merge_weight.max():.4f}, "
                  f"Mean: {self.merged_model.merge_weight.mean():.4f}")
        
        return self.merged_model.merge_weight.detach()
    
    def get_merged_model(self) -> nn.Module:
        """Get final merged model with task vectors baked in."""
        # Merge weights one final time
        self.merged_model.merge_weights()
        merged_backbone = self.merged_model.get_merged_backbone()
        
        class FinalMergedModel(nn.Module):
            def __init__(self, backbone, task_heads):
                super().__init__()
                self.backbone = backbone
                self.task_heads = nn.ModuleList(task_heads)
            
            def forward(self, input_ids, attention_mask=None, token_type_ids=None, task_idx: int = 0):
                backbone_output = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True
                )
                hidden_states = backbone_output.last_hidden_state
                logits = self.task_heads[task_idx](hidden_states[:, 0, :])
                
                from transformers.modeling_outputs import SequenceClassifierOutput
                return SequenceClassifierOutput(logits=logits)
        
        return FinalMergedModel(merged_backbone, self.merged_model.task_heads)
    
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
            agreement_count = 0
            total = 0
            start_time = time.time()
            
            with torch.no_grad():
                for batch in test_loader:
                    batch_device = {k: v.to(self.device) for k, v in batch.items() if v is not None}
                    
                    merged_output = merged_model(
                        input_ids=batch_device['input_ids'],
                        attention_mask=batch_device.get('attention_mask'),
                        token_type_ids=batch_device.get('token_type_ids'),
                        task_idx=task_idx
                    )
                    merged_preds = torch.argmax(merged_output.logits, dim=-1)
                    
                    expert_output = self.theta_t[task_idx](**batch_device)
                    expert_preds = torch.argmax(expert_output.logits, dim=-1)
                    
                    agreement_count += (merged_preds == expert_preds).sum().item()
                    total += merged_preds.shape[0]
            
            elapsed_time = time.time() - start_time
            agreement_rate = agreement_count / total if total > 0 else 0.0
            
            results[model_name] = {
                'agreement_rate': agreement_rate,
                'runtime_seconds': elapsed_time,
                'num_samples': total
            }
            
            print(f"Task '{model_name}': Agreement Rate={agreement_rate:.4f}, Runtime={elapsed_time:.2f}s")
        
        return results


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
    
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
        eta=0.001,
        num_epochs=5,
        batch_size=4,
        max_length=128,
        rho=0.07
    )
    
    print("=" * 60)
    print("Starting SAMerging Optimization...")
    print("=" * 60)

    optimal_weights = samerging.optimize_weights()
    merged_model = samerging.get_merged_model()

    print("\n" + "=" * 60)
    print("SAMerging completed!")
    print("=" * 60)
    print("\nOptimal merge weights shape:", optimal_weights.shape)
    print("Optimal merge weights:\n", optimal_weights)
    
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    test_results = samerging.evaluate_on_test_set(merged_model, dataset_dict, task_names)
    
    print("\n" + "=" * 60)
    print("Test Set Results Summary")
    print("=" * 60)
    
    total_agreement = 0.0
    
    for i, task_name in enumerate(task_names):
        model_key = list(dataset_dict.keys())[i]
        if model_key in test_results:
            results = test_results[model_key]
            print(f"\n{task_name}:")
            print(f"  Agreement Rate: {results['agreement_rate']:.4f}")
            print(f"  Runtime:        {results['runtime_seconds']:.4f}s")
            print(f"  # Samples:      {results['num_samples']}")
            total_agreement += results['agreement_rate']
    
    avg_agreement = total_agreement / len(task_names)
    
    print("\n" + "-" * 60)
    print(f"Average Agreement Rate across all tasks: {avg_agreement:.4f}")
    print("=" * 60)