# OLM Observer

**CHANGELOG:**
- ✅ Added complete OLM pipeline with P-C-D LSTM architecture
- ✅ Integrated State Predictor with temporal neural networks
- ✅ Added Hash Information Database for state tracking
- ✅ Enhanced GUI with prediction error display and thought/speech logs
- ✅ Updated engine with full sensory vector processing and internal state management
- ✅ **MAJOR FIXES**: Resolved critical runtime errors in D-LSTM training
- ✅ **PERFORMANCE**: Optimized energy consumption and novelty scaling
- ✅ **BEHAVIORAL**: Added comfort regeneration during reading cycles
- ✅ **CONSTRAINTS**: Fine-tuned reading triggers and cycle limits

OLM Observer is a comprehensive artificial consciousness simulation system that implements a complete Observational Learning Model (OLM) pipeline. The system captures, processes, and analyzes multimodal input streams including vision, keyboard interactions, and text input through a sophisticated neural architecture that includes pattern recognition, neurotransmitter simulation, and dual-output processing for internal thoughts and external speech.

## Features

### Core Components

- **Screen Capture**: Real-time screen capture using MSS (Multiple Screen Shots) with fallback to PyAutoGUI
- **Beta-VAE**: Variational Autoencoder for encoding visual data into latent space representations
- **Custom Tokenizer**: Dynamic vocabulary builder for text processing with persistent state
- **LSH System**: Locality-Sensitive Hashing for efficient similarity detection in high-dimensional spaces
- **Keyboard Monitoring**: Advanced keyboard event tracking with utterance packaging
- **Text Input Handler**: Queue-based text message processing system
- **GUI Interface**: Comprehensive Tkinter-based interface with real-time visualization

### Advanced Features

- **Multi-Modal Sensory Fusion**: Combines vision, text, and keyboard data into unified sensory packets
- **Real-Time Processing**: 20 TPS (Ticks Per Second) processing loop with performance monitoring
- **State Persistence**: Automatic checkpoint saving/loading for tokenizer and LSH memory
- **Performance Tracking**: Built-in TPS counter with visual feedback
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces

## System Requirements

### Python Dependencies

- Python 3.7+
- PyTorch 1.9.0+ (with torchvision)
- Pillow 8.0.0+
- NumPy 1.20.0+
- MSS 6.0.0+
- Pynput 1.7.0+
- Tkinter (usually included with Python)

### Hardware Requirements

- Multi-core CPU recommended for concurrent processing
- Minimum 4GB RAM (8GB+ recommended)
- Graphics card with CUDA support (optional, for GPU acceleration)
- Screen resolution: Any (automatically detected)

## Installation

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/A1CST/OLMV2.git
   cd OLMV2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python main.py
   ```

## Usage

### Basic Usage

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Use the GUI interface**
   - Click "Start" to begin data collection
   - Monitor real-time vision processing in the dual displays
   - Send text messages through the user input field
   - Monitor system logs in the bottom panels
   - Click "Stop" to halt processing
   - Use "Wipe" to clear all checkpoint data

### File Structure

```
OLMV2/
├── main.py              # Application entry point
├── engine.py            # Core processing engine with behavioral systems
├── gui.py               # Tkinter GUI interface with constraint monitoring
├── beta_vae.py          # Beta-VAE implementation
├── screen_capture.py    # Screen capture functionality
├── tokenizer.py         # Custom tokenizer
├── lsh_system.py        # LSH implementation
├── keyboard_input.py    # Keyboard monitoring
├── text_input.py        # Text input handling
├── prediction_model.py  # State prediction neural networks
├── neuro_predictor.py   # Neurotransmitter prediction networks
├── P_C_pipe.py          # Pattern recognition and neurotransmitter activation
├── D_pipe.py            # Dual D-LSTM for thought and speech generation
├── tinyllama_integration.py # External LLM integration
├── checkpoints/         # Persistent state storage
│   ├── lsh_memory.pkl   # LSH system state
│   ├── tokenizer_vocab.pkl # Tokenizer vocabulary
│   └── prediction_model.pth # State predictor weights
├── books/               # Reading material directory
├── logs/                # System reports and logs
├── dream_logs/          # Sleep cycle dream logs
└── requirements.txt     # Python dependencies
```

## Component Details

### Engine (engine.py)
The central processing unit that:
- Orchestrates all components in a 20 TPS loop
- Manages screen capture every 10 ticks
- Handles text and keyboard input processing
- Generates comprehensive sensory packets
- Implements LSH-based state hashing
- Provides checkpoint management

### Beta-VAE (beta_vae.py)
A frozen, untrained Variational Autoencoder that:
- Encodes 64x64 RGB images to 128-dimensional latent vectors
- Provides decode functionality for visualization
- Uses standard CNN architecture with ReLU activations
- Normalizes input using ImageNet statistics

### Screen Capture (screen_capture.py)
Robust screen capture system featuring:
- Primary MSS-based capture with automatic monitor detection
- PyAutoGUI fallback for compatibility
- Configurable bounding box support
- Error handling and blank image fallbacks

### Custom Tokenizer (tokenizer.py)
Dynamic vocabulary management system that:
- Builds vocabulary incrementally from input text
- Provides bidirectional token ↔ word mapping
- Supports persistent state via pickle checkpoints
- Handles unknown words with special tokens

### LSH System (lsh_system.py)
Advanced hashing system implementing:
- Random hyperplane-based LSH for vision data
- Multi-modal sensory packet hashing
- Hash frequency tracking and novelty detection
- State persistence and similarity metrics
- Master hash detection for convergence analysis

### Keyboard Input (keyboard_input.py)
Sophisticated keyboard monitoring featuring:
- Real-time key press/release tracking
- Hold duration calculation
- Utterance packaging with pause detection
- Thread-safe queue-based communication
- Clean shutdown handling

### GUI Interface (gui.py)
Comprehensive visualization system providing:
- Dual-panel vision display (input/prediction)
- Four-panel logging system (action/system/console/sensory)
- Real-time TPS monitoring
- User message input interface
- Control buttons for start/stop/wipe operations
- Console output redirection

## Configuration

### Engine Settings
- **Tick Rate**: 20 TPS (configurable in `engine.py`)
- **Vision Capture Frequency**: Every 10 ticks (2 Hz)
- **Model Save Frequency**: Every 200 ticks (10 seconds)

### Neural Network Parameters
- **Sensory Vector Size**: 195 (Vision:128 + Text:32 + Keyboard:3 + Previous Speech:32)
- **Prediction LSTM Hidden Size**: 256
- **Temporal Window**: 10 ticks for state prediction
- **Pattern LSTM Hidden Size**: 256, Output Size: 128
- **Neurotransmitter Vector Size**: 64
- **D-LSTM Output Tokens**: 32 maximum
- **Novelty Scaling Factor**: 10,000 (raw MSE → 0-100 behavioral range)

### LSH Parameters
- **Latent Dimension**: 128 (matches VAE output)
- **Number of Hashes**: 64 hyperplanes
- **Hash Type**: Binary string representation

### Behavioral System Parameters
- **Internal State Drivers**: Energy, Comfort, Novelty, Boredom (0-100 range)
- **Reading Constraints**: Boredom > 80, Comfort > 60, Novelty < 17.1
- **Reading Limit**: Maximum 1 book per awake cycle
- **Energy Consumption**: Speech (0.02/token), Thought (0.002), Decay (0.15)
- **Comfort Dynamics**: Decay (-0.02/tick wake), Regeneration (+0.05/tick reading)
- **Novelty Thresholds**: High (>100) resets boredom, Low (<50) increases boredom

### GUI Parameters
- **Window Size**: 1200x800 (resizable, minimum 800x600)
- **Log Panel Height**: 8 lines per panel (6 panels total)
- **Vision Display**: Auto-scaling with aspect ratio preservation
- **TPS Display**: Color-coded performance (Green/Yellow/Red)
- **Prediction Error**: Color-coded novelty visualization
- **Reading Constraints Display**: Real-time constraint monitoring with color coding

## Checkpoints and State Management

The system automatically saves state to the `checkpoints/` directory:

- **tokenizer_vocab.pkl**: Persistent vocabulary for the custom tokenizer
- **lsh_memory.pkl**: Hash frequency tracking and system memory

These files are created automatically and can be safely deleted to reset the system state.

## Performance Monitoring

The system includes built-in performance monitoring:

- **TPS Counter**: Real-time processing rate display
- **Target vs Actual**: Performance comparison metrics
- **Average TPS**: Rolling average calculation
- **Warning System**: Alerts when falling behind target rate

## Troubleshooting

### Common Issues

1. **Screen Capture Fails**
   - Ensure MSS is properly installed
   - Check screen resolution and permissions
   - System falls back to PyAutoGUI automatically

2. **High CPU Usage**
   - Reduce tick rate in `engine.py`
   - Increase vision capture interval
   - Close unnecessary applications

3. **Memory Issues**
   - Monitor checkpoint file sizes
   - Use "Wipe" button to clear accumulated data
   - Restart application periodically

4. **Keyboard Input Not Working**
   - Check pynput permissions
   - Ensure application has input accessibility rights
   - Verify keyboard listener startup messages

## Development Notes

### Architecture Principles
- **Modular Design**: Each component is self-contained and testable
- **Thread Safety**: Proper synchronization for concurrent operations
- **Error Resilience**: Comprehensive exception handling and fallbacks
- **State Persistence**: Automatic checkpoint management
- **Performance Monitoring**: Built-in metrics and optimization

### Extension Points
- **Custom VAE Models**: Replace beta_vae.py with trained models
- **Additional Input Sources**: Extend sensory packet structure
- **Alternative Hashing**: Implement different LSH configurations
- **ML Integration**: Add prediction and learning capabilities
- **Data Export**: Implement sensory data logging and analysis

## License

This project is provided as-is for educational and research purposes.

## Contributing

When modifying the codebase:
1. Maintain the existing component interfaces
2. Add appropriate error handling
3. Update checkpoint compatibility when changing data structures
4. Test GUI responsiveness under load
5. Document any new configuration parameters