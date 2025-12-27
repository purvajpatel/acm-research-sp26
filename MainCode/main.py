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


class TextDataset(Dataset):
    """Convert text dataframe to PyTorch dataset."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df.values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Handle different GLUE task formats
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


class SAMerging:
    """
    Sharpness-Aware Model Merging (SAMerging)
    Merges multiple fine-tuned models using layer-wise coefficients optimized via SAM.
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
        alpha_t: List[float] = None,
        batch_size: int = 4,
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            theta0: Pretrained base model
            theta_t: List of T fine-tuned models (one per task)
            calibration_datasets: Dict with structure {model_name: {data_type: dataframe}}
                                 (output from getDataset())
            tokenizer: HuggingFace tokenizer for text encoding
            rho: Neighborhood size for SAM
            eta: Learning rate for optimizer
            num_epochs: Total number of epochs
            alpha_t: Task loss weights (default: uniform 1/T)
            batch_size: Batch size for dataloaders
            max_length: Max token length for tokenizer
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        self.theta0 = theta0.to(device)
        self.theta_t = [model.to(device) for model in theta_t]
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.rho = rho
        self.eta = eta
        self.num_epochs = num_epochs
        self.T = len(theta_t)
        
        if alpha_t is None:
            self.alpha_t = [1.0 / self.T] * self.T
        else:
            self.alpha_t = alpha_t
        
        # Create dataloaders from GLUE datasets
        self.calibration_dataloaders = self._create_dataloaders(calibration_datasets)
        
        # Extract layer names and compute task vectors
        self.layer_names = self._get_layer_names()
        self.tau_t = self._compute_task_vectors()

        # Initialize lambda coefficients
        self._initialize_lambda_coeff()

        
    def _get_layer_names(self) -> List[str]:
        """Extract all layer names from the base model, excluding classifier."""
        layer_names = []
        for name, _ in self.theta0.named_parameters():
            # Skip classifier layer since models have different num_labels
            if 'classifier' in name:
                continue
            # Get layer name (remove .weight or .bias suffix)
            layer_name = '.'.join(name.split('.')[:-1])
            if layer_name not in layer_names:
                layer_names.append(layer_name)
        return layer_names
    
    def _compute_task_vectors(self) -> Dict[str, List[torch.Tensor]]:
        """
        Compute task vectors: τ_t = θ_t - θ_0
        Skips the classifier layer since models may have different num_labels.
        Returns: {layer_name: [tau_t1, tau_t2, ...]}
        """
        tau_t = {layer: [] for layer in self.layer_names}
        
        # Create param mapping for efficient access
        param_dict_0 = dict(self.theta0.named_parameters())
        
        for t, model_t in enumerate(self.theta_t):
            param_dict_t = dict(model_t.named_parameters())
            
            for layer in self.layer_names:
                # Find all parameters for this layer
                layer_delta = []
                for param_name, param_t in param_dict_t.items():
                    if param_name.startswith(layer):
                        param_0 = param_dict_0[param_name]
                        # Only include if shapes match
                        if param_t.shape == param_0.shape:
                            delta = (param_t - param_0).detach()
                            layer_delta.append(delta)
                
                # Concatenate all parameter deltas for this layer
                if layer_delta:
                    tau_t[layer].append(torch.cat([d.flatten() for d in layer_delta]))
        
        return tau_t
    
    def _create_dataloaders(self, calibration_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> List[DataLoader]:
        """
        Create dataloaders from calibration datasets.
        
        Args:
            calibration_datasets: Dict with structure {model_name: {data_type: dataframe}}
        
        Returns:
            List of dataloaders, one per task
        """
        dataloaders = []
        
        # Sort models to ensure consistent ordering
        models = sorted(calibration_datasets.keys())
        
        for model_name in models:
            # Get calibration/validation data (prefer 'validation' over others)
            if 'validation' in calibration_datasets[model_name]:
                df_cal = calibration_datasets[model_name]['validation']
            elif 'val' in calibration_datasets[model_name]:
                df_cal = calibration_datasets[model_name]['val']
            else:
                # Fallback to first available split
                df_cal = list(calibration_datasets[model_name].values())[0]
            
            # Create dataset and dataloader
            dataset = TextDataset(df_cal, self.tokenizer, self.max_length)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            dataloaders.append(dataloader)
        
        return dataloaders
    
    def _initialize_lambda_coeff(self):
        """Initialize lambda coefficients (one per task per layer)."""
        self.lambda_coeff = {
            layer: torch.ones(self.T, device=self.device, requires_grad=False) 
            for layer in self.layer_names
        }
    
    def construct_merged_model(self, lambda_coeff: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Construct merged model: θ_λ = θ_0 + Σ λ_t^l * τ_t^l
        """
        merged_model = self._clone_model(self.theta0)
        param_dict_0 = dict(self.theta0.named_parameters())
        param_dict_merged = dict(merged_model.named_parameters())
        
        for layer in self.layer_names:
            # Find all parameters for this layer
            param_offsets = {}
            offset = 0
            
            for param_name, param_0 in param_dict_0.items():
                if param_name.startswith(layer):
                    param_offsets[param_name] = (offset, offset + param_0.numel())
                    offset += param_0.numel()
            
            # Merge task vectors
            for param_name in param_offsets:
                # Check if parameter exists in merged model
                if param_name not in param_dict_merged:
                    continue
                    
                start, end = param_offsets[param_name]
                shape = param_dict_0[param_name].shape
                
                merged_delta = torch.zeros(end - start, device=self.device)
                for t in range(self.T):
                    if layer in lambda_coeff and t < len(self.tau_t[layer]):
                        merged_delta += lambda_coeff[layer][t] * self.tau_t[layer][t][start:end]
                
                param_dict_merged[param_name].data = (
                    param_dict_0[param_name] + merged_delta.reshape(shape)
                )
        
        return merged_model
    
    def knowledge_distillation_loss(
        self,
        theta_merge: nn.Module,
        temperature: float = 4.0
    ) -> torch.Tensor:
        """
        Calculate KL divergence loss between fine-tuned models (teachers)
        and merged model (student).
        """
        total_loss = 0.0
        
        for t in range(self.T):
            dataloader = self.calibration_dataloaders[t]
            task_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                # Move batch to device
                batch_device = {k: v.to(self.device) for k, v in batch.items() if v is not None}
                
                # Teacher model output
                with torch.no_grad():
                    teacher_output = self.theta_t[t](**batch_device)
                    # Extract logits from output object
                    if isinstance(teacher_output, tuple):
                        logits_teacher = teacher_output[0]
                    else:
                        logits_teacher = teacher_output.logits
                
                # Student model output
                student_output = theta_merge(**batch_device)
                # Extract logits from output object
                if isinstance(student_output, tuple):
                    logits_student = student_output[0]
                else:
                    logits_student = student_output.logits
                
                # Soft targets
                p_t = F.softmax(logits_teacher / temperature, dim=-1)
                q_lambda = F.log_softmax(logits_student / temperature, dim=-1)
                
                # KL divergence
                kl_loss = F.kl_div(q_lambda, p_t, reduction='batchmean')
                task_loss += kl_loss.item()
                num_batches += 1
            
            if num_batches > 0:
                task_loss /= num_batches
                total_loss += self.alpha_t[t] * task_loss
        
        return torch.tensor(total_loss, device=self.device, requires_grad=True)
    
    def optimize_lambda(self) -> Dict[str, torch.Tensor]:
        """
        Main optimization loop using SAM (Sharpness Aware Minimization).
        """
        # Flatten lambda coefficients for optimization
        lambda_vec = torch.cat([self.lambda_coeff[layer] for layer in self.layer_names])
        lambda_vec.requires_grad_(True)
        
        optimizer = Adam([lambda_vec], lr=self.eta)
        
        for epoch in range(self.num_epochs):
            # Unflatten lambda
            offset = 0
            lambda_dict = {}
            for layer in self.layer_names:
                lambda_dict[layer] = lambda_vec[offset:offset + self.T]
                offset += self.T
            
            # SAM Ascent Step: Find worst-case perturbation
            theta_lambda = self.construct_merged_model(lambda_dict)
            loss_kd = self.knowledge_distillation_loss(theta_lambda)
            
            # Compute gradient
            grad_lambda = torch.autograd.grad(loss_kd, lambda_vec, create_graph=True)[0]
            
            # Normalize and scale by rho
            grad_norm = torch.norm(grad_lambda)
            if grad_norm > 1e-8:
                epsilon = (self.rho * grad_lambda) / (grad_norm + 1e-8)
            else:
                epsilon = torch.zeros_like(grad_lambda)
            
            # SAM Descent Step: Update on perturbed parameters
            lambda_vec_perturbed = lambda_vec + epsilon
            lambda_vec_perturbed.requires_grad_(True)
            
            # Unflatten perturbed lambda
            offset = 0
            lambda_dict_perturbed = {}
            for layer in self.layer_names:
                lambda_dict_perturbed[layer] = lambda_vec_perturbed[offset:offset + self.T]
                offset += self.T
            
            theta_lambda_perturbed = self.construct_merged_model(lambda_dict_perturbed)
            loss_kd_perturbed = self.knowledge_distillation_loss(theta_lambda_perturbed)
            
            # Compute SAM gradient
            grad_sam = torch.autograd.grad(loss_kd_perturbed, lambda_vec_perturbed)[0]
            
            # Update coefficients
            optimizer.zero_grad()
            lambda_vec.grad = grad_sam
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss_kd.item():.4f}")
        
        # Return optimized coefficients
        offset = 0
        final_lambda = {}
        for layer in self.layer_names:
            final_lambda[layer] = lambda_vec.detach()[offset:offset + self.T]
            offset += self.T
        
        return final_lambda
    
    def get_merged_model(self) -> nn.Module:
        """
        Returns the final merged model.
        """
        return self.construct_merged_model(self.lambda_coeff)
    
    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)
    
    def evaluate_on_test_set(
        self,
        merged_model: nn.Module,
        test_datasets: Dict[str, Dict[str, pd.DataFrame]],
        task_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the merged model on test sets and compute accuracy per task.
        
        Args:
            merged_model: The merged model to evaluate
            test_datasets: Dict with structure {model_name: {data_type: dataframe}}
            task_names: List of task names (optional, for display purposes)
        
        Returns:
            Dict with accuracy and runtime for each task
        """
        results = {}
        
        # Sort models to ensure consistent ordering
        models = sorted(test_datasets.keys())
        
        for task_idx, model_name in enumerate(models):
            # Get test data
            if 'test' in test_datasets[model_name]:
                df_test = test_datasets[model_name]['test']
            elif 'test_matched' in test_datasets[model_name]:
                df_test = test_datasets[model_name]['test_matched']
            else:
                # Fallback to first available split
                df_test = list(test_datasets[model_name].values())[0]
            
            # Create test dataloader
            test_dataset = TextDataset(df_test, self.tokenizer, self.max_length)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Evaluate
            merged_model.eval()
            correct = 0
            total = 0
            start_time = time.time()
            
            with torch.no_grad():
                for batch in test_loader:
                    # Move batch to device
                    batch_device = {k: v.to(self.device) for k, v in batch.items() if v is not None}
                    
                    # Get predictions from merged model
                    outputs = merged_model(**batch_device)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Get ground truth (we need to add this back for evaluation)
                    # Note: We're using pseudo-labels from teacher models
                    with torch.no_grad():
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
    
    # Define paths to your local model directories (convert to forward slashes for cross-platform compatibility)
    base_model_path = os.path.abspath("./StudentModel").replace("\\", "/")  # θ₀: pretrained base model
    fine_tuned_model_paths = [
        os.path.abspath("./TeacherModels/CoLA").replace("\\", "/"),      # θ₁: fine-tuned on CoLA
        os.path.abspath("./TeacherModels/MNLI").replace("\\", "/"),      # θ₂: fine-tuned on MNLI
        os.path.abspath("./TeacherModels/MRPC").replace("\\", "/"),      # θ₃: fine-tuned on MRPC
        os.path.abspath("./TeacherModels/QNLI").replace("\\", "/"),      # θ₄: fine-tuned on QNLI
        os.path.abspath("./TeacherModels/QQP").replace("\\", "/"),       # θ₅: fine-tuned on QQP
        os.path.abspath("./TeacherModels/SST-2").replace("\\", "/")       # θ₆: fine-tuned on SST2
    ]
    
    task_names = ["CoLA", "MNLI", "MRPC", "QNLI", "QQP", "SST-2"]
    
    # Load tokenizer from the base model directory
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True, trust_remote_code=True)
    
# Load base model (θ₀) from local directory
    config_path = os.path.join(base_model_path, "config.json")
    config = BertConfig.from_json_file(config_path)
    # Keep the num_labels from the saved config file
    
    model_weights_path = os.path.join(base_model_path, "pytorch_model.bin")
    theta0 = BertForSequenceClassification(config)
    state_dict = torch.load(model_weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    theta0.load_state_dict(state_dict, strict=False)
    
    # Load fine-tuned models (θ_t) from local directories
    theta_t = []
    num_labels_per_task = []  # Track num_labels for each task
    
    for model_path in fine_tuned_model_paths:
        config_path = os.path.join(model_path, "config.json")
        config = BertConfig.from_json_file(config_path)
        # Keep the num_labels from the saved config file
        
        model_weights_path = os.path.join(model_path, "pytorch_model.bin")
        model = BertForSequenceClassification(config)
        state_dict = torch.load(model_weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict, strict=False)
        
        theta_t.append(model)
        num_labels_per_task.append(config.num_labels)

    from getDataset import getDataset
    
    # Load GLUE datasets
    dataset_dict = getDataset()

    
    # Initialize SAMerging
    samerging = SAMerging(
        theta0=theta0,
        theta_t=theta_t,
        calibration_datasets=dataset_dict,
        tokenizer=tokenizer,
        rho=0.05,
        eta=0.01,
        num_epochs=5,
        batch_size=4,
        max_length=128
    )
    
    # Run optimization
    print("=" * 60)
    print("Starting SAMerging Optimization...")
    print("=" * 60)

    optimal_lambda = samerging.optimize_lambda()
    samerging.lambda_coeff = optimal_lambda  # Store the optimized coefficients
    merged_model = samerging.get_merged_model()

    print("\n" + "=" * 60)
    print("SAMerging completed!")
    print("=" * 60)
    print("\nOptimal lambda coefficients:")
    for layer_name, coeffs in list(optimal_lambda.items())[:3]:  # Show first 3 layers
        print(f"  {layer_name}: {coeffs.tolist()}")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    test_results = samerging.evaluate_on_test_set(merged_model, dataset_dict, task_names)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Set Results Summary")
    print("=" * 60)
    
    total_accuracy = 0.0
    total_runtime = 0.0
    
    for i, task_name in enumerate(task_names):
        model_key = list(dataset_dict.keys())[i]
        if model_key in test_results:
            results = test_results[model_key]
            print(f"\n{task_name}:")
            print(f"  Accuracy:    {results['accuracy']:.4f}")
            print(f"  Runtime:     {results['runtime_seconds']:.4f}s")
            print(f"  # Samples:   {results['num_samples']}")
            total_accuracy += results['accuracy']
            total_runtime += results['runtime_seconds']
    
    avg_accuracy = total_accuracy / len(task_names)
    avg_runtime = total_runtime / len(task_names)
    
    print("\n" + "-" * 60)
    print(f"Average Accuracy across all tasks: {avg_accuracy:.4f}")
    print(f"Average Runtime per task:         {avg_runtime:.4f}s")
    print(f"Total Runtime:                    {total_runtime:.4f}s")
    print("=" * 60)