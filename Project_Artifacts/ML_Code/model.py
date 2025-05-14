#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model architecture for molecular solubility prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import config


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, embedding_dim, max_seq_length=config.MAX_SEQ_LENGTH):
        """
        Initialize positional encoding
        
        Args:
            embedding_dim: Embedding dimension
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to avoid counting as model parameters
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings [batch_size, seq_length, embedding_dim]
            
        Returns:
            Embeddings with positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class SmilesTransformerEncoder(nn.Module):
    """Transformer encoder for SMILES strings"""
    
    def __init__(
        self,
        vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        feedforward_dim=config.FEEDFORWARD_DIM,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dropout_attn=config.DROPOUT_ATTN,
        dropout_ff=config.DROPOUT_FF
    ):
        """
        Initialize SMILES transformer encoder
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout_attn: Dropout rate in attention layers
            dropout_ff: Dropout rate in feed-forward layers
        """
        super().__init__()
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_attn)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout_ff,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Encoded representation [batch_size, seq_length, embedding_dim]
        """
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for transformer
        # (1 = not masked, 0 = masked)
        if attention_mask is not None:
            # Convert boolean mask to float and invert for transformer
            # (transformer expects 0 for tokens to attend to, 1 for tokens to ignore)
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
            
        # Apply transformer encoder
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=attention_mask)
        
        return encoded


class MolecularDescriptorEmbedding(nn.Module):
    """Embedding layer for molecular descriptors"""
    
    def __init__(
        self, 
        num_descriptors: int = len(config.MOLECULAR_DESCRIPTORS),
        hidden_dim: int = 64,
        embedding_dim: int = config.EMBEDDING_DIM
    ):
        """
        Initialize descriptor embedding
        
        Args:
            num_descriptors: Number of descriptors
            hidden_dim: Hidden dimension
            embedding_dim: Final embedding dimension
        """
        super().__init__()
        
        self.descriptor_projection = nn.Sequential(
            nn.Linear(num_descriptors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Project descriptors to embedding space
        
        Args:
            descriptors: Molecular descriptors [batch_size, num_descriptors]
            
        Returns:
            Descriptor embeddings [batch_size, embedding_dim]
        """
        return self.descriptor_projection(descriptors)


class SolubilityTransformer(nn.Module):
    """Transformer model for molecular solubility prediction"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        num_layers: int = config.NUM_LAYERS,
        num_heads: int = config.NUM_HEADS,
        feedforward_dim: int = config.FEEDFORWARD_DIM,
        max_seq_length: int = config.MAX_SEQ_LENGTH,
        num_descriptors: int = len(config.MOLECULAR_DESCRIPTORS),
        dropout_attn: float = config.DROPOUT_ATTN,
        dropout_ff: float = config.DROPOUT_FF
    ):
        """
        Initialize solubility transformer model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            feedforward_dim: Feed-forward dimension
            max_seq_length: Maximum sequence length
            num_descriptors: Number of molecular descriptors
            dropout_attn: Dropout rate in attention layers
            dropout_ff: Dropout rate in feed-forward layers
        """
        super().__init__()
        
        # SMILES transformer encoder
        self.smiles_encoder = SmilesTransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            max_seq_length=max_seq_length,
            dropout_attn=dropout_attn,
            dropout_ff=dropout_ff
        )
        
        # Molecular descriptor embedding
        self.descriptor_embedding = MolecularDescriptorEmbedding(
            num_descriptors=num_descriptors,
            hidden_dim=64,
            embedding_dim=embedding_dim
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout_ff),
            nn.Linear(feedforward_dim, 1)
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        descriptor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            descriptor: Molecular descriptors [batch_size, num_descriptors]
            
        Returns:
            Tuple of (prediction, hidden_states)
        """
        # Encode SMILES
        encoded = self.smiles_encoder(input_ids, attention_mask)
        
        # Get CLS token representation (first token)
        cls_token = encoded[:, 0, :]
        
        # Get descriptor embedding
        desc_embedding = self.descriptor_embedding(descriptor)
        
        # Concatenate CLS token and descriptor embedding
        combined = torch.cat([cls_token, desc_embedding], dim=1)
        
        # Apply regression head
        prediction = self.regression_head(combined)
        
        # Extract hidden states for interpretability
        hidden_states = {
            "cls_token": cls_token,
            "desc_embedding": desc_embedding,
            "combined": combined,
            "encoded": encoded
        }
        
        return prediction, hidden_states
