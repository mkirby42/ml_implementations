import torch
import torch.nn as nn

class FrequencyPredictor(nn.Module):
    def __init__(self, encoder, input_channels=64, input_length=375):
        super().__init__()
        self.encoder = encoder
        
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.flatten_dim = input_channels * input_length
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Ensure input has channel dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Get embeddings from frozen encoder
        with torch.no_grad():
            embeddings = self.encoder(x)
            
        return self.head(embeddings)

