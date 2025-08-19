import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Neuro vector configuration
NEUROTRANSMITTER_VECTOR_SIZE = 64  # Size of neuro vectors from the neuro activator
PREDICTION_HIDDEN_SIZE = 256
TEMPORAL_WINDOW_SIZE = 10


class NeuroPredictionModel(nn.Module):
    """
    Neural network module for predicting the next neuro state vector.
    Uses LSTM to learn temporal patterns in neuro data.
    """
    
    def __init__(self, input_size=NEUROTRANSMITTER_VECTOR_SIZE, hidden_size=PREDICTION_HIDDEN_SIZE):
        """
        Initialize the prediction model with LSTM and linear layers.
        
        Args:
            input_size (int): Size of the input neuro vector
            hidden_size (int): Size of the LSTM hidden state
        """
        super(NeuroPredictionModel, self).__init__()
        
        # LSTM layer for temporal pattern learning
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Linear output layer to map back to neuro vector space
        self.output_layer = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        """
        Forward pass through the prediction model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Predicted next neuro vector of shape (batch_size, input_size)
        """
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the output from the last time step
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through linear output layer
        prediction = self.output_layer(last_output)  # Shape: (batch_size, input_size)
        
        return prediction


class NeuroPredictor:
    """
    Handler class for managing the neuro prediction model and neuro buffer.
    Responsible for maintaining temporal context and making predictions.
    """
    
    def __init__(self, device=None):
        """
        Initialize the neuro predictor.
        
        Args:
            device (str, optional): PyTorch device ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize the prediction model
        self.model = NeuroPredictionModel()
        self.model.to(self.device)
        
        # Initialize neuro buffer for temporal context
        self.sensory_buffer = deque(maxlen=TEMPORAL_WINDOW_SIZE)
        
        # Set model to evaluation mode by default
        self.model.eval()
    
    def add_to_buffer(self, neuro_vector):
        """
        Add a neuro vector to the buffer for training.
        
        Args:
            neuro_vector (numpy.ndarray): Neuro vector to add to buffer
        """
        self.sensory_buffer.append(neuro_vector)
    
    def predict(self, current_neuro_vector):
        """
        Predict the next neuro state based on current and historical data.
        
        Args:
            current_neuro_vector (numpy.ndarray): Current neuro vector
            
        Returns:
            numpy.ndarray: Predicted next neuro vector
        """
        # Add current vector to buffer
        self.sensory_buffer.append(current_neuro_vector)
        
        # If we don't have enough data, return the current vector
        if len(self.sensory_buffer) < 2:
            return current_neuro_vector
        
        # Convert buffer to tensor
        buffer_list = list(self.sensory_buffer)
        sequence = np.array(buffer_list)
        
        # Add batch dimension and convert to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
        
        # Convert back to numpy and return
        return prediction.cpu().numpy()[0]
    
    def train(self, input_sequence, target_sequence):
        """
        Train the model on a sequence of neuro vectors.
        
        Args:
            input_sequence (numpy.ndarray): Input sequence of neuro vectors
            target_sequence (numpy.ndarray): Target sequence of neuro vectors
            
        Returns:
            float: Training loss
        """
        # Set model to training mode
        self.model.train()
        
        # Convert to tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor(target_sequence, dtype=torch.float32, device=self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Forward pass
        prediction = self.model(input_tensor)
        loss = criterion(prediction, target_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Set model back to evaluation mode
        self.model.eval()
        
        return loss.item()
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        """
        Load the model from a file.
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

