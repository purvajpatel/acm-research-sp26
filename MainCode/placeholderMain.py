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


class BackboneMergedModel(nn.Module):
    """
    Merges BERT backbones while keeping task-specific heads separate.
    
    Key idea:
    - During optimization: apply weighted task vectors on-the-fly via temporary parameter modifications
    - Each task keeps its own classification head
    - After optimization: create final merged model with task vectors baked in
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        finetuned_models: List[nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.base_model = base_model
        self.finetuned_models = nn.ModuleList(finetuned_models)
        self.device = device
        self.num_tasks = len(finetuned_models)
        
        # Extract backbones and heads
        self.backbone = self._extract_backbone(base_model)
        self.task_heads = nn.ModuleList([self._extract_head(model) for model in finetuned_models])
        self.finetuned_backbones = nn.ModuleList([self._extract_backbone(model) for model in finetuned_models])
        
        # Count layers in backbone
        self.num_layers = self._count_layers(self.backbone)
        
        # Layer-wise merge weights: (num_tasks, num_layers)
        self.register_parameter('merge_weight', nn.Parameter(
            torch.ones(self.num_tasks, self.num_layers, device=device) / self.num_tasks
        ))
        
        # Compute task vectors for backbone only (not head)
        self.task_vectors = self._compute_task_vectors()
    
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
    
    def _compute_task_vectors(self) -> List[Dict[str, torch.Tensor]]:
        """Compute task vectors as: θ_t[backbone] - θ_0[backbone]"""
        task_vectors = []
        
        base_backbone_params = dict(self.backbone.named_parameters())
        
        for ft_backbone in self.finetuned_backbones:
            ft_backbone_params = dict(ft_backbone.named_parameters())
            
            task_vector_dict = {}
            for param_name, base_param in base_backbone_params.items():
                if param_name in ft_backbone_params:
                    ft_param = ft_backbone_params[param_name]
                    if base_param.shape == ft_param.shape:
                        task_vector_dict[param_name] = (ft_param - base_param).detach()
            
            task_vectors.append(task_vector_dict)
        
        return task_vectors
    
    def _apply_task_vectors(self, merge_weight: torch.Tensor):
        """Apply weighted task vectors to backbone parameters."""
        base_params = dict(self.backbone.named_parameters())
        
        for param_name in base_params:
            weighted_tv = None
            
            for task_idx in range(self.num_tasks):
                if param_name in self.task_vectors[task_idx]:
                    tv = self.task_vectors[task_idx][param_name].to(self.device)
                    
                    # Get layer index
                    layer_idx = self._extract_layer_idx(param_name)
                    if layer_idx is not None:
                        weight = merge_weight[task_idx, layer_idx]
                    else:
                        weight = merge_weight[task_idx].mean()
                    
                    if weighted_tv is None:
                        weighted_tv = weight * tv
                    else:
                        weighted_tv = weighted_tv + weight * tv
            
            # Apply to backbone (without no_grad to allow gradient flow)
            if weighted_tv is not None:
                base_params[param_name].data.add_(weighted_tv)
    
    def _unapply_task_vectors(self, merge_weight: torch.Tensor):
        """Remove task vectors from backbone parameters."""
        base_params = dict(self.backbone.named_parameters())
        
        for param_name in base_params:
            weighted_tv = None
            
            for task_idx in range(self.num_tasks):
                if param_name in self.task_vectors[task_idx]:
                    tv = self.task_vectors[task_idx][param_name].to(self.device)
                    
                    layer_idx = self._extract_layer_idx(param_name)
                    if layer_idx is not None:
                        weight = merge_weight[task_idx, layer_idx]
                    else:
                        weight = merge_weight[task_idx].mean()
                    
                    if weighted_tv is None:
                        weighted_tv = weight * tv
                    else:
                        weighted_tv = weighted_tv + weight * tv
            
            if weighted_tv is not None:
                with torch.no_grad():
                    base_params[param_name].data.sub_(weighted_tv)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, task_idx: int = 0):
        """Forward pass with merged backbone (if merging is enabled)."""
        backbone_output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        hidden_states = backbone_output.last_hidden_state
        task_head = self.task_heads[task_idx]
        logits = task_head(hidden_states[:, 0, :])
        
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(logits=logits)
    
    def get_merged_backbone(self) -> nn.Module:
        """Create final merged backbone with task vectors baked in."""
        merged_backbone = copy.deepcopy(self.backbone)
        
        with torch.no_grad():
            base_params = dict(self.backbone.named_parameters())
            
            for param_name, base_param in base_params.items():
                weighted_tv = None
                
                for task_idx in range(self.num_tasks):
                    if param_name in self.task_vectors[task_idx]:
                        tv = self.task_vectors[task_idx][param_name].to(self.device)
                        
                        layer_idx = self._extract_layer_idx(param_name)
                        if layer_idx is not None:
                            weight = self.merge_weight[task_idx, layer_idx]
                        else:
                            weight = self.merge_weight[task_idx].mean()
                        
                        if weighted_tv is None:
                            weighted_tv = weight * tv
                        else:
                            weighted_tv = weighted_tv + weight * tv
                
                if weighted_tv is not None:
                    for name, param in merged_backbone.named_parameters():
                        if name == param_name:
                            param.data.add_(weighted_tv)
                            break
        
        return merged_backbone
    
    def _extract_layer_idx(self, param_name: str) -> Optional[int]:
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


class SAMerging:
    """
    Sharpness-Aware Model Merging for multi-task BERT models.
    
    Implements the SAMerging algorithm with:
    - Multi-teacher KL divergence loss
    - Sharpness-Aware Minimization (SAM) optimization
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
        self.merged_model = BackboneMergedModel(theta0, theta_t, device).to(device)
        
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
        total_loss = None
        models = sorted(self.calibration_datasets.keys())
        alpha = 1.0 / self.T
        
        for task_idx, model_name in enumerate(models):
            batch = next(self.batch_iterators[model_name])
            batch_device = {k: v.to(self.device) for k, v in batch.items() if v is not None}
            
            # Forward through merged backbone + task head
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
            
            # KL divergence: KL(p || q) = Σ p log(p/q) = Σ p log(p) - Σ p log(q)
            # Using PyTorch's F.kl_div which computes KL(target || input)
            # where input is log-probabilities and target is probabilities
            kl_loss = F.kl_div(q_student, p_teacher, reduction='batchmean')
            
            if total_loss is None:
                total_loss = alpha * kl_loss
            else:
                total_loss = total_loss + alpha * kl_loss
        
        return total_loss
    
    def optimize_weights(self) -> torch.Tensor:
        """
        Optimize merge weights using Sharpness-Aware Minimization (SAM).
        
        SAM procedure:
        1. Ascent step: find worst-case perturbation ε within neighborhood ρ
        2. Descent step: compute gradient at perturbed point
        3. Update: apply gradient update and remove perturbation
        """
        # Only optimize merge_weight parameter
        params_to_optimize = [self.merged_model.merge_weight]
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.eta)
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # ===== SAM Ascent Step =====
            # Apply task vectors with current weights
            self.merged_model._apply_task_vectors(self.merged_model.merge_weight)
            
            # Compute loss at current point
            loss = self.compute_kl_loss()
            loss.backward()
            
            # Get gradient for normalization
            if self.merged_model.merge_weight.grad is not None:
                grad = self.merged_model.merge_weight.grad.clone()
                grad_norm = torch.norm(grad) + 1e-12
                
                # Compute perturbation: ε = ρ * g / ||g||_2
                epsilon = (self.rho / grad_norm) * grad
                
                # ===== SAM Descent Step =====
                # Remove task vectors before perturbing weights
                self.merged_model._unapply_task_vectors(self.merged_model.merge_weight)
                
                # Apply perturbation to weights
                with torch.no_grad():
                    self.merged_model.merge_weight.data.add_(epsilon)
                
                # Reapply task vectors with perturbed weights
                self.merged_model._apply_task_vectors(self.merged_model.merge_weight)
                
                # Compute loss at perturbed point
                optimizer.zero_grad()
                loss_perturbed = self.compute_kl_loss()
                loss_perturbed.backward()
                
                # Remove perturbation and task vectors (back to original point)
                self.merged_model._unapply_task_vectors(self.merged_model.merge_weight)
                with torch.no_grad():
                    self.merged_model.merge_weight.data.sub_(epsilon)
                
                # Reapply task vectors for next iteration
                self.merged_model._apply_task_vectors(self.merged_model.merge_weight)
            else:
                loss_perturbed = loss
            
            # Update weights using gradient from perturbed point
            optimizer.step()
            
            # Log progress
            grad_norm = self.merged_model.merge_weight.grad.norm().item() if self.merged_model.merge_weight.grad is not None else 0.0
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}, "
                  f"Gradient norm: {grad_norm:.6f}, "
                  f"Weights - Min: {self.merged_model.merge_weight.min():.4f}, "
                  f"Max: {self.merged_model.merge_weight.max():.4f}, "
                  f"Mean: {self.merged_model.merge_weight.mean():.4f}")
        
        return self.merged_model.merge_weight.detach()
    
    def get_merged_model(self) -> nn.Module:
        """Get final merged model with task vectors baked in."""
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