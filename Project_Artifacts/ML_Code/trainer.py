#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training and evaluation utilities for molecular solubility prediction
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Union
import logging
import json
from pathlib import Path
import config
from model import SolubilityTransformer
from data_processing import SolubilityDataset


class SolubilityTrainer:
    """Trainer for molecular solubility prediction model"""
    
    def __init__(
        self,
        model: SolubilityTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        warmup_ratio: float = config.WARMUP_RATIO,
        num_epochs: int = config.NUM_EPOCHS,
        patience: int = config.PATIENCE,
        device: Optional[torch.device] = None,
        model_dir: str = str(config.MODEL_DIR),
        results_dir: str = str(config.RESULTS_DIR)
    ):
        """
        Initialize trainer
        
        Args:
            model: Solubility transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_ratio: Portion of steps for learning rate warmup
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            device: Device to use for training
            model_dir: Directory to save models
            results_dir: Directory to save results
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Initialize LR scheduler
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer, 
            warmup_steps, 
            total_steps
        )
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.results_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_rmse = []
        self.val_mae = []
        self.val_r2 = []
        self.learning_rates = []
        
    def _get_cosine_schedule_with_warmup(
        self, 
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Create a cosine learning rate schedule with warmup
        
        Args:
            optimizer: Optimizer
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            num_cycles: Number of cycles for cosine annealing
            last_epoch: Last epoch
            
        Returns:
            Learning rate scheduler
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * num_cycles * 2.0 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def train_epoch(self) -> float:
        """
        Train model for one epoch
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            descriptor = batch["descriptor"].to(self.device)
            solubility = batch["solubility"].to(self.device).view(-1, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            prediction, _ = self.model(input_ids, attention_mask, descriptor)
            
            # Calculate loss
            loss = self.criterion(prediction, solubility)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * input_ids.size(0)
            
        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader.dataset)
        
        # Track learning rate
        self.learning_rates.append(self.scheduler.get_last_lr()[0])
        
        return avg_loss
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on data loader
        
        Args:
            data_loader: Data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                descriptor = batch["descriptor"].to(self.device)
                solubility = batch["solubility"].to(self.device).view(-1, 1)
                
                # Forward pass
                prediction, _ = self.model(input_ids, attention_mask, descriptor)
                
                # Calculate loss
                loss = self.criterion(prediction, solubility)
                
                # Update metrics
                total_loss += loss.item() * input_ids.size(0)
                
                # Collect predictions and targets
                all_predictions.extend(prediction.cpu().numpy())
                all_targets.extend(solubility.cpu().numpy())
                
        # Calculate metrics
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        
        # Calculate MAE
        mae = mean_absolute_error(all_targets, all_predictions)
        
        # Calculate R²
        r2 = r2_score(all_targets, all_predictions)
        
        # Calculate Pearson correlation
        pearson = np.corrcoef(all_targets, all_predictions)[0, 1]
        
        return {
            "loss": avg_loss,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pearson": pearson,
            "predictions": all_predictions,
            "targets": all_targets
        }
    
    def train(self) -> Dict[str, List]:
        """
        Train model for multiple epochs
        
        Returns:
            Dictionary of training metrics
        """
        best_val_rmse = float('inf')
        best_epoch = 0
        no_improve_count = 0
        start_time = time.time()
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader)
            self.val_losses.append(val_metrics["loss"])
            self.val_rmse.append(val_metrics["rmse"])
            self.val_mae.append(val_metrics["mae"])
            self.val_r2.append(val_metrics["r2"])
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val RMSE: {val_metrics['rmse']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"Val R²: {val_metrics['r2']:.4f}, "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )
            
            # Check for improvement
            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_epoch = epoch
                no_improve_count = 0
                
                # Save best model
                self.save_model("best_model.pt")
                self.logger.info(f"Saved best model (RMSE: {best_val_rmse:.4f})")
            else:
                no_improve_count += 1
                
            # Check for early stopping
            if no_improve_count >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        # Log training summary
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch+1}")
        
        # Save final model
        self.save_model("final_model.pt")
        
        # Save training metrics
        self.save_metrics()
        
        # Plot training curves
        self.plot_training_curves()
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_rmse": self.val_rmse,
            "val_mae": self.val_mae,
            "val_r2": self.val_r2,
            "learning_rates": self.learning_rates
        }
    
    def test(self) -> Dict[str, float]:
        """
        Test model on test set
        
        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            self.logger.warning("No test loader provided")
            return {}
        
        # Load best model
        self.load_model("best_model.pt")
        
        # Evaluate on test set
        test_metrics = self.evaluate(self.test_loader)
        
        # Log test metrics
        self.logger.info(
            f"Test RMSE: {test_metrics['rmse']:.4f}, "
            f"Test MAE: {test_metrics['mae']:.4f}, "
            f"Test R²: {test_metrics['r2']:.4f}, "
            f"Test Pearson: {test_metrics['pearson']:.4f}"
        )
        
        # Save test metrics
        test_metrics_dict = {
            "rmse": float(test_metrics["rmse"]),
            "mae": float(test_metrics["mae"]),
            "r2": float(test_metrics["r2"]),
            "pearson": float(test_metrics["pearson"])
        }
        
        with open(os.path.join(self.results_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics_dict, f, indent=4)
            
        # Plot test predictions
        self.plot_predictions(
            test_metrics["predictions"],
            test_metrics["targets"],
            "test_predictions.png"
        )
        
        return test_metrics
    
    def save_model(self, filename: str) -> None:
        """
        Save model to disk
        
        Args:
            filename: Model filename
        """
        model_path = os.path.join(self.model_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, model_path)
    
    def load_model(self, filename: str) -> None:
        """
        Load model from disk
        
        Args:
            filename: Model filename
        """
        model_path = os.path.join(self.model_dir, filename)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    def save_metrics(self) -> None:
        """Save training metrics to disk"""
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_rmse": self.val_rmse,
            "val_mae": self.val_mae,
            "val_r2": self.val_r2,
            "learning_rates": self.learning_rates
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for k, v in metrics.items():
            metrics[k] = [float(x) for x in v]
            
        with open(os.path.join(self.results_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    
    def plot_training_curves(self) -> None:
        """Plot training curves"""
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label="Train")
        plt.plot(self.val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        
        # Plot RMSE
        plt.subplot(2, 2, 2)
        plt.plot(self.val_rmse)
        plt.axhline(y=config.TARGET_RMSE, color='r', linestyle='--', label=f"Target ({config.TARGET_RMSE})")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Validation RMSE")
        plt.legend()
        
        # Plot MAE
        plt.subplot(2, 2, 3)
        plt.plot(self.val_mae)
        plt.axhline(y=config.TARGET_MAE, color='r', linestyle='--', label=f"Target ({config.TARGET_MAE})")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Validation MAE")
        plt.legend()
        
        # Plot R²
        plt.subplot(2, 2, 4)
        plt.plot(self.val_r2)
        plt.axhline(y=config.TARGET_R2, color='r', linestyle='--', label=f"Target ({config.TARGET_R2})")
        plt.xlabel("Epoch")
        plt.ylabel("R²")
        plt.title("Validation R²")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_curves.png"))
        plt.close()
        
    def plot_predictions(self, predictions: np.ndarray, targets: np.ndarray, filename: str) -> None:
        """
        Plot predictions against targets
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            filename: Output filename
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate error metrics for title
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # Plot predictions vs targets
        plt.scatter(targets, predictions, alpha=0.5)
        
        # Plot ideal line
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Labels and title
        plt.xlabel("Measured Solubility (log units)")
        plt.ylabel("Predicted Solubility (log units)")
        plt.title(f"Solubility Predictions (RMSE: {rmse:.3f}, R²: {r2:.3f})")
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add error bands
        error_bands = [0.5, 1.0, 1.5]
        for band in error_bands:
            plt.fill_between(
                [min_val, max_val],
                [min_val - band, max_val - band],
                [min_val + band, max_val + band],
                alpha=0.1,
                color='g'
            )
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename))
        plt.close()


def analyze_model_errors(
    model: SolubilityTransformer,
    test_loader: DataLoader,
    tokenizer,
    descriptor_calculator,
    device: torch.device = None
) -> pd.DataFrame:
    """
    Analyze model errors
    
    Args:
        model: Trained model
        test_loader: Test data loader
        tokenizer: SMILES tokenizer
        descriptor_calculator: Molecular descriptor calculator
        device: Device to use
        
    Returns:
        DataFrame with error analysis
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.to(device)
    model.eval()
    
    # Collect predictions and errors
    predictions = []
    targets = []
    errors = []
    smiles_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get batch data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            descriptor = batch["descriptor"].to(device)
            solubility = batch["solubility"].to(device).view(-1, 1)
            
            # Get predictions
            prediction, _ = model(input_ids, attention_mask, descriptor)
            
            # Collect data
            predictions.extend(prediction.cpu().numpy().flatten())
            targets.extend(solubility.cpu().numpy().flatten())
            errors.extend((prediction - solubility).cpu().numpy().flatten())
            
            # Decode SMILES from input_ids
            for ids in input_ids.cpu().numpy():
                smiles = tokenizer.decode(ids)
                smiles_list.append(smiles)
    
    # Create DataFrame
    df = pd.DataFrame({
        "SMILES": smiles_list,
        "Measured": targets,
        "Predicted": predictions,
        "Error": errors,
        "AbsError": np.abs(errors)
    })
    
    # Calculate molecular descriptors for all SMILES
    descriptor_df = descriptor_calculator.calculate_batch(df["SMILES"])
    
    # Combine dataframes
    result_df = pd.concat([df, descriptor_df], axis=1)
    
    # Calculate additional statistics
    result_df["ErrorCategory"] = pd.cut(
        result_df["AbsError"],
        bins=[0, 0.5, 1.0, 1.5, float('inf')],
        labels=["<0.5", "0.5-1.0", "1.0-1.5", ">1.5"]
    )
    
    return result_df


def perform_y_randomization_test(
    model_class: Callable,
    train_dataset: SolubilityDataset,
    val_dataset: SolubilityDataset,
    batch_size: int = config.BATCH_SIZE,
    num_trials: int = 5,
    device: torch.device = None,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform Y-randomization test to confirm model learns meaningful patterns
    
    Args:
        model_class: Model class constructor
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_trials: Number of trials
        device: Device to use
        **model_kwargs: Keyword arguments for model constructor
        
    Returns:
        Dictionary of metrics for each trial
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    results = {
        "original_rmse": [],
        "random_rmse": [],
        "original_r2": [],
        "random_r2": []
    }
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Train original model
    original_model = model_class(**model_kwargs)
    trainer = SolubilityTrainer(
        model=original_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=10  # Reduced epochs for quicker testing
    )
    trainer.train()
    val_metrics = trainer.evaluate(val_loader)
    
    results["original_rmse"].append(val_metrics["rmse"])
    results["original_r2"].append(val_metrics["r2"])
    
    # Perform Y-randomization trials
    for trial in range(num_trials):
        # Create randomized dataset
        random_train_dataset = SolubilityDataset(
            train_dataset.smiles,
            np.random.permutation(train_dataset.solubility),
            train_dataset.tokenizer,
            train_dataset.descriptor_calculator,
            augment=False
        )
        
        random_train_loader = DataLoader(
            random_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Train randomized model
        random_model = model_class(**model_kwargs)
        random_trainer = SolubilityTrainer(
            model=random_model,
            train_loader=random_train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=10  # Reduced epochs for quicker testing
        )
        random_trainer.train()
        random_val_metrics = random_trainer.evaluate(val_loader)
        
        results["random_rmse"].append(random_val_metrics["rmse"])
        results["random_r2"].append(random_val_metrics["r2"])
        
    return results
