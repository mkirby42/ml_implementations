import torch
import torch.nn as nn

class SignalReconstructor(nn.Module):
    def __init__(self, input_channels=1, base_filters=16):
        super().__init__()
        
        # Encoder: Compresses the signal
        # Input: (B, 1, 3000)
        self.encoder = nn.Sequential(
            # Layer 1: 3000 -> 1500
            nn.Conv1d(input_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            
            # Layer 2: 1500 -> 750
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            
            # Layer 3: 750 -> 375
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(),
        )
        
        # Decoder: Reconstructs the signal
        self.decoder = nn.Sequential(
            # Layer 1: 375 -> 750
            nn.ConvTranspose1d(base_filters * 4, base_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            
            # Layer 2: 750 -> 1500
            nn.ConvTranspose1d(base_filters * 2, base_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            
            # Layer 3: 1500 -> 3000
            nn.ConvTranspose1d(base_filters, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # No activation or Tanh/Sigmoid depending on data range. 
            # Since data is likely centered around 0 with large amplitude, linear is safest unless normalized.
        )

    def forward(self, x):
        # Expect input shape: (batch_size, seq_len) or (batch_size, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Return (batch_size, seq_len)
        return decoded.squeeze(1)

