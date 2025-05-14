#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for molecular solubility prediction using transformer models
"""

import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import config
from data_processing import (
    prepare_data,
    create_dataloaders
)
from model import SolubilityTransformer
from trainer import (
    SolubilityTrainer,
    analyze_model_errors,
    perform_y_randomization_test
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Molecular solubility prediction using transformers")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the solubility dataset CSV file")
    parser.add_argument("--smiles_col", type=str, default="SMILES",
                        help="Column name for SMILES strings in the dataset")
    parser.add_argument("--solubility_col", type=str, default="Solubility",
                        help="Column name for solubility values in the dataset")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict", "analysis"],
                        help="Operation mode: train, evaluate, predict, or analysis")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=config.NUM_EPOCHS,
                        help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=config.NUM_LAYERS,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=config.NUM_HEADS,
                        help="Number of attention heads")
    parser.add_argument("--embedding_dim", type=int, default=config.EMBEDDING_DIM,
                        help="Embedding dimension")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to saved model for evaluation or prediction")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    
    # Set output directory
    if args.output_dir:
        config.RESULTS_DIR = Path(args.output_dir)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Load and prepare data
    print(f"Loading data from {args.data_path}")
    train_dataset, val_dataset, test_dataset, tokenizer, descriptor_calculator = prepare_data(
        data_path=args.data_path,
        smiles_col=args.smiles_col,
        solubility_col=args.solubility_col,
        augment_train=(args.mode == "train")
    )
    
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = SolubilityTransformer(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize trainer
    trainer = SolubilityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=device
    )
    
    # Execute based on mode
    if args.mode == "train":
        print("Starting training...")
        trainer.train()
        print("Evaluating on test set...")
        test_metrics = trainer.test()
        print(f"Test metrics: {test_metrics}")
        
    elif args.mode == "eval":
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(config.MODEL_DIR, "best_model.pt")
            
        print(f"Loading model from {model_path}")
        trainer.load_model(model_path)
        
        print("Evaluating on test set...")
        test_metrics = trainer.test()
        print(f"Test metrics: {test_metrics}")
        
    elif args.mode == "predict":
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(config.MODEL_DIR, "best_model.pt")
            
        print(f"Loading model from {model_path}")
        trainer.load_model(model_path)
        
        print("Generating predictions...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            "SMILES": [tokenizer.decode(ids.tolist()) for ids in test_dataset.smiles],
            "Measured": test_metrics["targets"],
            "Predicted": test_metrics["predictions"],
            "Error": test_metrics["predictions"] - test_metrics["targets"]
        })
        
        predictions_path = os.path.join(config.RESULTS_DIR, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")
        
    elif args.mode == "analysis":
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(config.MODEL_DIR, "best_model.pt")
            
        print(f"Loading model from {model_path}")
        trainer.load_model(model_path)
        
        print("Analyzing model errors...")
        error_df = analyze_model_errors(
            model=model,
            test_loader=test_loader,
            tokenizer=tokenizer,
            descriptor_calculator=descriptor_calculator,
            device=device
        )
        
        error_path = os.path.join(config.RESULTS_DIR, "error_analysis.csv")
        error_df.to_csv(error_path, index=False)
        print(f"Error analysis saved to {error_path}")
        
        # Plot error distribution by descriptor values
        print("Generating error analysis plots...")
        for descriptor in config.MOLECULAR_DESCRIPTORS:
            if descriptor in error_df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(error_df[descriptor], error_df["AbsError"], alpha=0.5)
                plt.xlabel(descriptor)
                plt.ylabel("Absolute Error")
                plt.title(f"Error vs {descriptor}")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(config.RESULTS_DIR, f"error_vs_{descriptor}.png"))
                plt.close()
                
        # Perform Y-randomization test
        print("Performing Y-randomization test...")
        y_rand_results = perform_y_randomization_test(
            model_class=SolubilityTransformer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            vocab_size=len(tokenizer.vocab),
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads
        )
        
        # Save Y-randomization results
        y_rand_df = pd.DataFrame(y_rand_results)
        y_rand_path = os.path.join(config.RESULTS_DIR, "y_randomization_results.csv")
        y_rand_df.to_csv(y_rand_path, index=False)
        print(f"Y-randomization results saved to {y_rand_path}")
        
        # Plot Y-randomization results
        plt.figure(figsize=(10, 6))
        plt.bar(
            ['Original RMSE', 'Random RMSE', 'Original R²', 'Random R²'],
            [
                y_rand_results['original_rmse'][0],
                sum(y_rand_results['random_rmse']) / len(y_rand_results['random_rmse']),
                y_rand_results['original_r2'][0],
                sum(y_rand_results['random_r2']) / len(y_rand_results['random_r2'])
            ]
        )
        plt.ylabel('Value')
        plt.title('Y-Randomization Test Results')
        plt.savefig(os.path.join(config.RESULTS_DIR, "y_randomization.png"))
        plt.close()
        
    print("Done!")


if __name__ == "__main__":
    main()
