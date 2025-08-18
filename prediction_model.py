import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Sensory vector configuration
SENSORY_VECTOR_SIZE = 163  # Composition: Vision (128) + Text (32) + Keyboard (3)
MAX_TEXT_TOKENS = 32
PREDICTION_HIDDEN_SIZE = 256
TEMPORAL_WINDOW_SIZE = 10


class PredictionModel(nn.Module):
    """
    Neural network module for predicting the next sensory state vector.
    Uses LSTM to learn temporal patterns in sensory data.
    """
    
    def __init__(self, input_size=SENSORY_VECTOR_SIZE, hidden_size=PREDICTION_HIDDEN_SIZE):
        """
        Initialize the prediction model with LSTM and linear layers.
        
        Args:
            input_size (int): Size of the input sensory vector
            hidden_size (int): Size of the LSTM hidden state
        """
        super(PredictionModel, self).__init__()
        
        # LSTM layer for temporal pattern learning
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Linear output layer to map back to sensory vector space
        self.output_layer = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        """
        Forward pass through the prediction model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Predicted next sensory vector of shape (batch_size, input_size)
        """
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the output from the last time step
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through linear output layer
        prediction = self.output_layer(last_output)  # Shape: (batch_size, input_size)
        
        return prediction


class StatePredictor:
    """
    Handler class for managing the prediction model and sensory buffer.
    Responsible for maintaining temporal context and making predictions.
    """
    
    def __init__(self, device=None):
        """
        Initialize the state predictor.
        
        Args:
            device (str, optional): PyTorch device ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize the prediction model
        self.model = PredictionModel()
        self.model.to(self.device)
        
        # Initialize sensory buffer for temporal context
        self.sensory_buffer = deque(maxlen=TEMPORAL_WINDOW_SIZE)
        
        # Set model to evaluation mode by default
        self.model.eval()
    
    def predict(self, current_sensory_vector):
        """
        Predict the next sensory state based on current and historical data.
        
        Args:
            current_sensory_vector (numpy.ndarray): Current sensory vector
            
        Returns:
            numpy.ndarray: Predicted next sensory vector
        """
        # Convert numpy array to torch tensor
        current_tensor = torch.tensor(current_sensory_vector, dtype=torch.float32, device=self.device)
        
        # Append to sensory buffer
        self.sensory_buffer.append(current_tensor)
        
        # Check if buffer is full
        if len(self.sensory_buffer) < TEMPORAL_WINDOW_SIZE:
            # Buffer not full - pad with zeros
            padding_size = TEMPORAL_WINDOW_SIZE - len(self.sensory_buffer)
            padded_buffer = [torch.zeros_like(current_tensor) for _ in range(padding_size)]
            padded_buffer.extend(list(self.sensory_buffer))
            buffer_tensor = torch.stack(padded_buffer)
        else:
            # Buffer is full - use as is
            buffer_tensor = torch.stack(list(self.sensory_buffer))
        
        # Add batch dimension
        batched_input = buffer_tensor.unsqueeze(0)  # Shape: (1, sequence_length, input_size)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(batched_input)
        
        # Convert back to numpy array and flatten
        predicted_vector = prediction.squeeze(0).cpu().numpy()
        
        return predicted_vector
    
    def train(self):
        """
        Placeholder for training method.
        Training logic will be implemented later.
        """
        pass
    
    def save_model(self, filepath):
        """
        Save the prediction model to disk.
        
        Args:
            filepath (str): Path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': SENSORY_VECTOR_SIZE,
                'hidden_size': PREDICTION_HIDDEN_SIZE
            }
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load the prediction model from disk.
        
        Args:
            filepath (str): Path to the saved model file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


if __name__ == "__main__":
    # Demonstration usage
    print("Creating StatePredictor...")
    predictor = StatePredictor()
    
    # Create dummy sensory vectors
    print("\nTesting prediction with dummy data...")
    dummy_vector = np.random.randn(SENSORY_VECTOR_SIZE)
    
    # Make a few predictions to fill the buffer
    for i in range(5):
        prediction = predictor.predict(dummy_vector)
        print(f"Prediction {i+1}: Shape = {prediction.shape}, Mean = {prediction.mean():.3f}")
        
        # Update dummy vector for next iteration
        dummy_vector = np.random.randn(SENSORY_VECTOR_SIZE)
    
    print(f"\nBuffer size: {len(predictor.sensory_buffer)}")
    print(f"Model device: {predictor.device}")
    print("Prediction model demonstration completed successfully!")
