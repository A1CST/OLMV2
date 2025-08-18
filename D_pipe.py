import torch
import torch.nn as nn
import numpy as np

# Placeholder for the max number of tokens the D-LSTMs can output
MAX_OUTPUT_TOKENS = 32

class BaseD_LSTM(nn.Module):
    """A base class for a D-LSTM to define the dynamic paths."""
    def __init__(self, output_size):
        super(BaseD_LSTM, self).__init__()
        # A list of layers to be executed sequentially
        self.layers = nn.ModuleList([
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            # Final output layer
            nn.Linear(512, output_size)
        ])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, neuro_vector, depth=None):
        # If depth is not specified or is too large, use all layers
        if depth is None or depth > len(self.layers):
            depth = len(self.layers)

        x = torch.tensor(neuro_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Process the input through the layers up to the specified depth
        for i in range(depth):
            x = self.layers[i](x)

        return x

class ThoughtD_LSTM:
    def __init__(self):
        self.model = BaseD_LSTM(output_size=MAX_OUTPUT_TOKENS)
        self.model.eval()

    def process(self, neuro_vector, internal_state, depth=None):
        with torch.no_grad():
            output_tensor = self.model(neuro_vector, depth=depth)
        # Return the vector as a numpy array
        return output_tensor.squeeze(0).cpu().numpy()

class TextD_LSTM:
    def __init__(self):
        self.model = BaseD_LSTM(output_size=MAX_OUTPUT_TOKENS)
        self.model.eval()

    def process(self, neuro_vector, internal_state, depth=None):
        with torch.no_grad():
            output_tensor = self.model(neuro_vector, depth=depth)
        # Return the vector as a numpy array
        return output_tensor.squeeze(0).cpu().numpy()
