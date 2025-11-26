import torch
import torch.nn as nn

class AutoencoderBasedSignalReconstructor(nn.Module):
    def __init__(self, input_channels=1, base_filters=16):
        super().__init__()
        
        self.base_filters = base_filters
        
        # Encoder layers (separate for skip connections)
        # Layer 1: 3000 -> 1500
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
        )
        
        # Layer 2: 1500 -> 750
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
        )
        
        # Layer 3: 750 -> 375
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(),
        )
        
        # Decoder layers (input channels doubled due to skip concatenation)
        # Layer 1: 375 -> 750
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base_filters * 4, base_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
        )
        
        # Layer 2: 750 -> 1500 (input: dec1 output + enc2 skip = base_filters * 4)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_filters * 4, base_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
        )
        
        # Layer 3: 1500 -> 3000 (input: dec2 output + enc1 skip = base_filters * 2)
        self.dec3 = nn.ConvTranspose1d(base_filters * 2, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        """Run encoder and return bottleneck features."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return e3

    def forward(self, x):
        # Expect input shape: (batch_size, seq_len) or (batch_size, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Encoder with saved activations for skip connections
        e1 = self.enc1(x)    # (B, 16, 1500)
        e2 = self.enc2(e1)   # (B, 32, 750)
        e3 = self.enc3(e2)   # (B, 64, 375)
        
        # Decoder with skip connections (concatenation)
        d1 = self.dec1(e3)                    # (B, 32, 750)
        d1 = torch.cat([d1, e2], dim=1)       # (B, 64, 750)
        
        d2 = self.dec2(d1)                    # (B, 16, 1500)
        d2 = torch.cat([d2, e1], dim=1)       # (B, 32, 1500)
        
        d3 = self.dec3(d2)                    # (B, 1, 3000)
        
        # Return (batch_size, seq_len)
        return d3.squeeze(1)
    
    @property
    def encoder(self):
        """For compatibility with FrequencyPredictor - returns a callable that produces encoder output."""
        return self.encode

