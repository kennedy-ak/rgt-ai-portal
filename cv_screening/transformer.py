import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class ApplicantTransformer(nn.Module):
    def __init__(self, categorical_dim, bert_dim=768, num_classes=5):
        super(ApplicantTransformer, self).__init__()

        # Dimensions
        # Size of categorical features after preprocessing
        self.categorical_dim = categorical_dim
        # Size of BERT embeddings (768 for base BERT)
        self.bert_dim = bert_dim
        self.hidden_dim = 256
        # Number of possible stages (1st interview, Technical, etc.)
        self.num_classes = num_classes

        # Embedding layers
        self.categorical_fc = nn.Linear(categorical_dim, 128)
        self.bert_fc = nn.Linear(bert_dim, 128)

        # Transformer encoder for feature interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,      # Dimension of input embeddings
            nhead=4,          # Number of attention heads
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)

        # Output head for score prediction (regression)
        self.score_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Output head for stage prediction (classification)
        self.stage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, categorical_features, text_embeddings):
        # Process categorical features
        cat_emb = F.relu(self.categorical_fc(categorical_features))

        # Process BERT embeddings
        text_emb = F.relu(self.bert_fc(text_embeddings))

        # Combine features for transformer input
        # Shape: [2, batch_size, embedding_dim]
        combined = torch.stack([cat_emb, text_emb], dim=0)

        # Apply transformer to learn feature interactions
        transformed = self.transformer_encoder(combined)

        # Pool the output by concatenating transformed representations
        pooled = torch.cat([
            transformed[0],  # Transformed categorical features
            transformed[1]   # Transformed text embeddings
        ], dim=1)

        # Generate predictions
        score = self.score_head(pooled)  # Applicant score prediction
        stage = self.stage_head(pooled)  # Furthest stage prediction

        return score, stage

# Function to create PyTorch data loaders


def create_data_loaders(categorical_features, text_embeddings, scores, stages, batch_size=16):
    # Convert to PyTorch tensors
    cat_tensor = torch.FloatTensor(categorical_features)
    text_tensor = torch.FloatTensor(text_embeddings)
    score_tensor = torch.FloatTensor(scores).view(-1, 1)
    stage_tensor = torch.LongTensor(stages)

    # Create TensorDataset
    dataset = TensorDataset(cat_tensor, text_tensor,
                            score_tensor, stage_tensor)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader