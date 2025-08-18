import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Note: The following are placeholders for a future config file
SENSORY_VECTOR_SIZE = 163
PATTERN_LSTM_HIDDEN_SIZE = 256
PATTERN_LSTM_OUTPUT_SIZE = 128
LSTM1_TEMPORAL_WINDOW_SIZE = 10
NEURO_LSTM_HIDDEN_SIZE = 128
NEUROTRANSMITTER_VECTOR_SIZE = 64
PATTERN_VECTOR_SIZE = PATTERN_LSTM_OUTPUT_SIZE
LSTM2_TEMPORAL_WINDOW_SIZE = 5

# --- P-LSTM (Pattern Recognizer) ---
class PatternLSTMModel(nn.Module):
    def __init__(self):
        super(PatternLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=SENSORY_VECTOR_SIZE, hidden_size=PATTERN_LSTM_HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(PATTERN_LSTM_HIDDEN_SIZE, PATTERN_LSTM_OUTPUT_SIZE)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.output_layer(lstm_out[:, -1, :])

class PatternRecognizer:
    def __init__(self):
        self.model = PatternLSTMModel()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.sensory_buffer = deque(maxlen=LSTM1_TEMPORAL_WINDOW_SIZE)

    def process(self, sensory_input_vector): # MODIFIED: Takes vector as argument
        # Note: Novelty logic from old file is omitted for this placeholder
        sensory_input = torch.tensor(sensory_input_vector, dtype=torch.float32, device=self.device)
        self.sensory_buffer.append(sensory_input)
        
        if len(self.sensory_buffer) < LSTM1_TEMPORAL_WINDOW_SIZE:
            padding_needed = LSTM1_TEMPORAL_WINDOW_SIZE - len(self.sensory_buffer)
            padded_buffer = [torch.zeros_like(sensory_input)] * padding_needed + list(self.sensory_buffer)
            batched_input = torch.stack(padded_buffer)
        else:
            batched_input = torch.stack(list(self.sensory_buffer))
        
        batched_input = batched_input.unsqueeze(0)

        with torch.no_grad():
            pattern_output = self.model(batched_input)

        return pattern_output.cpu().numpy().flatten().astype(np.float32)

# --- C-LSTM (Neurotransmitter Activator) ---
class NeurotransmitterLSTMModel(nn.Module):
    def __init__(self):
        super(NeurotransmitterLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=PATTERN_VECTOR_SIZE, hidden_size=NEURO_LSTM_HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(NEURO_LSTM_HIDDEN_SIZE, NEUROTRANSMITTER_VECTOR_SIZE)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.output_layer(lstm_out[:, -1, :])

class NeurotransmitterActivator:
    def __init__(self):
        self.model = NeurotransmitterLSTMModel()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pattern_buffer = deque(maxlen=LSTM2_TEMPORAL_WINDOW_SIZE)

    def process(self, pattern_input_vector): # MODIFIED: Takes vector as argument
        pattern_input = torch.tensor(pattern_input_vector, dtype=torch.float32, device=self.device)
        self.pattern_buffer.append(pattern_input)
        
        if len(self.pattern_buffer) < LSTM2_TEMPORAL_WINDOW_SIZE:
            padding_needed = LSTM2_TEMPORAL_WINDOW_SIZE - len(self.pattern_buffer)
            padded_buffer = [torch.zeros_like(pattern_input)] * padding_needed + list(self.pattern_buffer)
            batched_input = torch.stack(padded_buffer)
        else:
            batched_input = torch.stack(list(self.pattern_buffer))
        
        batched_input = batched_input.unsqueeze(0)

        with torch.no_grad():
            neurotransmitter_output = self.model(batched_input)

        return neurotransmitter_output.cpu().numpy().flatten().astype(np.float32)
