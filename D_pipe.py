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
        
        # Setup for training
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, neuro_vector, depth=None):
        # If depth is not specified or is too large, use all layers
        if depth is None or depth > len(self.layers):
            depth = len(self.layers)

        x = torch.tensor(neuro_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Process the input through the layers up to the specified depth
        for i in range(depth):
            x = self.layers[i](x)

        return x
        
    def train_step(self, input_vector, target_vector, depth):
        """Performs a single training step."""
        self.train() # Switch to training mode

        # Prepare tensors - ensure input matches expected size
        if len(input_vector) != 64:
            # Resize input to match the first layer's expected input size
            if len(input_vector) > 64:
                input_vector = input_vector[:64]  # Truncate
            else:
                # Pad with zeros if too small
                padded_input = np.zeros(64, dtype=np.float32)
                padded_input[:len(input_vector)] = input_vector
                input_vector = padded_input
                
        # Ensure target matches the output size of the model
        output_size = MAX_OUTPUT_TOKENS  # This is the final output size
        if len(target_vector) != output_size:
            if len(target_vector) > output_size:
                target_vector = target_vector[:output_size]  # Truncate
            else:
                # Pad with zeros if too small
                padded_target = np.zeros(output_size, dtype=np.float32)
                padded_target[:len(target_vector)] = target_vector
                target_vector = padded_target
                
        target_tensor = torch.tensor(target_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward pass - always use full depth for training to get final output
        self.optimizer.zero_grad()
        prediction = self.forward(input_vector, depth=None)  # Use full depth

        # Calculate loss and update weights
        loss = self.criterion(prediction, target_tensor)
        loss.backward()
        self.optimizer.step()

        self.eval() # Switch back to evaluation mode
        return loss.item()

    def get_max_depth(self):
        """Returns the total number of layers in the model."""
        return len(self.layers)

class ThoughtD_LSTM:
    def __init__(self):
        self.model = BaseD_LSTM(output_size=MAX_OUTPUT_TOKENS)
        self.model.eval()

    def process(self, neuro_vector, internal_state, depth=None):
        with torch.no_grad():
            output_tensor = self.model(neuro_vector, depth=depth)
        # Return the vector as a numpy array
        return output_tensor.squeeze(0).cpu().numpy()

    def get_max_depth(self):
        """Returns the maximum processing depth of the underlying model."""
        return self.model.get_max_depth()
        
    def train(self, input_vector, target_vector, depth):
        """Triggers a training step in the base model."""
        return self.model.train_step(input_vector, target_vector, depth)

class TextD_LSTM:
    def __init__(self):
        self.model = BaseD_LSTM(output_size=MAX_OUTPUT_TOKENS)
        self.model.eval()

    def process(self, neuro_vector, internal_state, depth=None):
        with torch.no_grad():
            output_tensor = self.model(neuro_vector, depth=depth)
        # Return the vector as a numpy array
        return output_tensor.squeeze(0).cpu().numpy()

    def get_max_depth(self):
        """Returns the maximum processing depth of the underlying model."""
        return self.model.get_max_depth()
        
    def train(self, input_vector, target_vector, depth):
        """Triggers a training step in the base model."""
        return self.model.train_step(input_vector, target_vector, depth)
