import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DisasterRiskTransformer(nn.Module):
    """
    Transformer-based time-series model for predicting disaster risk probabilities.
    Inputs: Concatenated features from weather, environmental indicators, and detection outputs.
    """
    def __init__(
        self, 
        input_dim: int, 
        d_model: int = 64, 
        nhead: int = 4, 
        num_layers: int = 3, 
        dropout: float = 0.1,
        output_dim: int = 1 # Probability of disaster risk
    ):
        super(DisasterRiskTransformer, self).__init__()
        self.model_type = 'Transformer'
        
        # Linear layer to project input features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.d_model = d_model
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.Sigmoid() # Output as probability
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        # Linear layers inside decoder
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Tensor of shape [seq_len, batch_size, input_dim]
            src_mask: Optional mask for the sequence
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        
        # Use the representation of the last time step for risk prediction
        # Alternatively, use global average pooling over the sequence
        last_step = output[-1, :, :]
        return self.decoder(last_step)
