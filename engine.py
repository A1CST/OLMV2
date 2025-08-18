import threading
import time
import os
import numpy as np
from screen_capture import ScreenCapturer
from beta_vae import BetaVAE
from text_input import TextInputHandler
from keyboard_input import KeyboardListener
from tokenizer import CustomTokenizer
from lsh_system import LSHSystem
from prediction_model import StatePredictor, MAX_TEXT_TOKENS
from P_C_pipe import PatternRecognizer, NeurotransmitterActivator
from D_pipe import ThoughtD_LSTM, TextD_LSTM
from hash_info_db import HashInfoDB

class Engine:
    def __init__(self, gui=None):
        """Initialize the engine with a reference to the GUI"""
        self.gui = gui
        self.tick_rate = 20  # 20 ticks per second
        self.running = False
        self.thread = None
        self.checkpoints_dir = "checkpoints"
        
        # Create checkpoints directory if it doesn't exist
        self.ensure_checkpoints_dir()
        
        # Initialize screen capturer
        self.capturer = ScreenCapturer()
        
        # Initialize Beta-VAE model
        self.vae = BetaVAE()
        
        # Initialize text input handler
        self.text_handler = TextInputHandler()
        
        # Initialize keyboard listener
        self.keyboard_listener = KeyboardListener()
        
        # Initialize tokenizer with checkpoint file
        tokenizer_checkpoint_file = os.path.join(self.checkpoints_dir, "tokenizer_vocab.pkl")
        self.tokenizer = CustomTokenizer(checkpoint_path=tokenizer_checkpoint_file)
        
        # Initialize LSH system with checkpoint file
        lsh_checkpoint_file = os.path.join(self.checkpoints_dir, "lsh_memory.pkl")
        self.lsh_system = LSHSystem(latent_dim=128, num_hashes=64, checkpoint_path=lsh_checkpoint_file)
        
        # Initialize the hash information database
        hash_db_checkpoint_file = os.path.join(self.checkpoints_dir, "hash_info_db.pkl")
        self.hash_db = HashInfoDB(checkpoint_path=hash_db_checkpoint_file)
        
        # Initialize persistent sensory packet
        self.current_sensory_packet = {}
        
        # Initialize hash tracking
        self.last_tick_hash = None
        
        # Initialize the state prediction model
        self.predictor = StatePredictor()
        
        # Load the prediction model if a checkpoint exists
        predictor_checkpoint_file = os.path.join(self.checkpoints_dir, "prediction_model.pth")
        if os.path.exists(predictor_checkpoint_file):
            try:
                self.predictor.load_model(predictor_checkpoint_file)
                if self.gui:
                    self.gui.log_system("Loaded prediction model from checkpoint.")
                else:
                    print("Loaded prediction model from checkpoint.")
            except Exception as e:
                error_msg = f"Error loading prediction model: {e}"
                if self.gui:
                    self.gui.log_system(error_msg)
                else:
                    print(error_msg)
        
        # Track the previous sensory vector for training purposes
        self.previous_sensory_vector = None
        
        # Store the prediction from the previous tick for error calculation
        self.prediction_from_previous_tick = None
        
        # Initialize the main OLM pipeline models
        self.pattern_recognizer = PatternRecognizer()
        self.neuro_activator = NeurotransmitterActivator()
        self.thought_d_lstm = ThoughtD_LSTM()
        self.text_d_lstm = TextD_LSTM()
        
        # Initialize internal state
        self.internal_state = {
            'Novelty': 0.0,
            'Energy': 100.0,
            'Boredom': 0.0,
            'Comfort': 75.0
        }
        
        # Track the state before an action is taken
        self.state_before_action = None
        
    def _create_sensory_vector(self, sensory_packet):
        """Convert sensory packet dictionary into a fixed-length sensory vector"""
        # Initialize zero vectors for each sensory modality
        vision_vector = np.zeros(128)
        text_vector = np.zeros(MAX_TEXT_TOKENS)
        keyboard_vector = np.zeros(3)
        
        # Process vision data
        if 'vision' in sensory_packet and 'latent_vector' in sensory_packet['vision']:
            vision_vector = sensory_packet['vision']['latent_vector']
        
        # Process text data
        if 'text' in sensory_packet:
            tokens = sensory_packet['text']['text']
            if isinstance(tokens, list):
                # Truncate or pad the token list to MAX_TEXT_TOKENS
                if len(tokens) > MAX_TEXT_TOKENS:
                    text_vector = np.array(tokens[:MAX_TEXT_TOKENS])
                else:
                    text_vector[:len(tokens)] = tokens
        
        # Process keyboard data
        if 'keyboard' in sensory_packet:
            keyboard_data = sensory_packet['keyboard']
            events = keyboard_data.get('events', [])
            
            # Calculate summary statistics
            num_events = len(events)
            total_hold_duration = 0.0
            
            # Correctly iterate through the list of event dictionaries
            for event in events:
                if isinstance(event, dict) and event.get('type') == 'hold':
                    total_hold_duration += event.get('hold_duration', 0.0)
            
            # Use the accurate total_duration from the keyboard_utterance packet
            total_utterance_duration = keyboard_data.get('total_duration', 0.0)
            
            keyboard_vector[0] = num_events
            keyboard_vector[1] = total_hold_duration
            keyboard_vector[2] = total_utterance_duration
        
        # Concatenate all vectors into the final sensory vector
        sensory_vector = np.concatenate([vision_vector, text_vector, keyboard_vector])
        
        return sensory_vector
        
    def start(self):
        """Start the engine in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.daemon = True
            self.thread.start()
            
            # Start the keyboard listener
            self.keyboard_listener.start()
            
            if self.gui:
                self.gui.log_system("Engine starting...")
                self.gui.reset_tps_counter()
            else:
                print("Engine starting...")
                
    def stop(self):
        """Stop the engine"""
        if self.running:
            # Stop the keyboard listener first for clean shutdown
            self.keyboard_listener.stop()
            
            # Save LSH state before stopping
            if self.gui:
                self.gui.log_system("Saving LSH memory...")
            else:
                print("Saving LSH memory...")
            self.lsh_system.save_state()
            
            # Save tokenizer state before stopping
            if self.gui:
                self.gui.log_system("Saving tokenizer vocabulary...")
            else:
                print("Saving tokenizer vocabulary...")
            self.tokenizer.save_state()
            
            # Save prediction model state before stopping
            if self.gui:
                self.gui.log_system("Saving prediction model state...")
            else:
                print("Saving prediction model state...")
            predictor_checkpoint_file = os.path.join(self.checkpoints_dir, "prediction_model.pth")
            self.predictor.save_model(predictor_checkpoint_file)
            
            # Save HashInfoDB state
            if self.gui:
                self.gui.log_system("Saving Hash Info DB...")
            else:
                print("Saving Hash Info DB...")
            self.hash_db.save_state()
            
            self.running = False
            
            if self.gui:
                self.gui.log_system("Engine stopping...")
            else:
                print("Engine stopping...")
                
    def _run_loop(self):
        """Main engine loop running in a separate thread"""
        tick_count = 0
        
        while self.running:
            # Get start time for this tick
            start_time = time.time()
            
            # Check for new text input
            text_input = self.text_handler.get_latest_input()
            if text_input is not None:
                # Tokenize the text input
                original_text = text_input['text']
                tokenized_text = self.tokenizer.tokenize(original_text)
                
                # Replace raw text with tokenized version
                text_input['text'] = tokenized_text
                
                if self.gui:
                    self.gui.log_system(f"Text input received from [{text_input['source']}]: '{original_text}' -> tokens: {tokenized_text}")
                else:
                    print(f"Text input received from [{text_input['source']}]: '{original_text}' -> tokens: {tokenized_text}")
                # TODO: Package text_input into the main sensory snapshot
            
            # Check for new keyboard input
            keyboard_utterance = self.keyboard_listener.get_latest_utterance()
            if keyboard_utterance is not None:
                if self.gui:
                    self.gui.log_system(f"Keyboard utterance received with {len(keyboard_utterance['events'])} events.")
                else:
                    print(f"Keyboard utterance received with {len(keyboard_utterance['events'])} events.")
                # TODO: Package keyboard_utterance into the main sensory snapshot
            
            # Check if we should capture screen (every 10 ticks)
            vision_data = None
            if tick_count % 10 == 0:
                try:
                    # Capture screen frame
                    captured_image = self.capturer.capture_frame()
                    
                    # Log capture info for debugging
                    if self.gui:
                        self.gui.log_system(f"Captured screen at tick {tick_count} - Size: {captured_image.size}")
                    
                    # Process the captured frame through the VAE
                    if self.gui:
                        self.gui.log_system("Encoding captured frame...")
                    
                    # Encode the image to latent space
                    latent_vector = self.vae.encode(captured_image)
                    
                    if self.gui:
                        self.gui.log_system(f"Latent vector created with shape: {latent_vector.shape}")
                    
                    # Update GUI with captured frame
                    if self.gui:
                        self.gui.update_vision(captured_image)
                    
                    # Store vision data for sensory packet
                    vision_data = {
                        'latent_vector': latent_vector,
                        'image_size': captured_image.size
                    }
                        
                except Exception as e:
                    error_msg = f"Error capturing screen at tick {tick_count}: {str(e)}"
                    if self.gui:
                        self.gui.log_system(error_msg)
                    else:
                        print(error_msg)
            
            # Accumulate sensory data in persistent packet
            # Add vision data if available (from screen capture)
            if vision_data is not None:
                self.current_sensory_packet['vision'] = vision_data
            
            # Add text input if available
            if text_input is not None:
                self.current_sensory_packet['text'] = text_input
                if self.gui:
                    self.gui.log_system(f"Added text to sensory packet: '{text_input['text']}'")
            
            # Add keyboard input if available
            if keyboard_utterance is not None:
                self.current_sensory_packet['keyboard'] = keyboard_utterance
                if self.gui:
                    self.gui.log_system(f"Added keyboard utterance to sensory packet: {len(keyboard_utterance['events'])} events")
            
            # Generate LSH hash for the current sensory packet
            if self.current_sensory_packet:
                current_hash = self.lsh_system.generate_hash(self.current_sensory_packet, self.last_tick_hash)
                
                # Log the new hash to the system log
                if self.gui:
                    self.gui.log_system(f"Generated LSH hash: {current_hash[:16]}...")
                
                # Update the last hash for the next tick
                self.last_tick_hash = current_hash
                
                # Add the current hash to the sensory packet
                self.current_sensory_packet['lsh_hash'] = current_hash
            
            # --- Prediction, Error Calculation, and Visualization ---

            # 1. Create the sensory vector for the current state
            current_sensory_vector = self._create_sensory_vector(self.current_sensory_packet)

            # 2. If a prediction from the last tick exists, calculate the error
            if self.prediction_from_previous_tick is not None:
                # Calculate Mean Squared Error between the last prediction and the current actual vector
                prediction_error = np.mean((self.prediction_from_previous_tick - current_sensory_vector)**2)

                # Log the error, which represents novelty
                if self.gui:
                    self.gui.log_system(f"Prediction Error (Novelty): {prediction_error:.6f}")
                    # Update the GUI's prediction error display
                    self.gui.update_prediction_error(prediction_error)
                
                # Update internal state with the novelty value
                self.internal_state['Novelty'] = prediction_error

                # TODO: Use this error to train the predictor and influence Drivers
                # self.predictor.train(self.previous_sensory_vector, current_sensory_vector)
            else:
                # No prediction available yet, update GUI to show no data
                if self.gui:
                    self.gui.update_prediction_error(None)

            # 3. Predict the sensory vector for the *next* tick based on the *current* vector
            predicted_next_vector = self.predictor.predict(current_sensory_vector)

            # 4. Decode the vision part of the new prediction for visualization
            if self.gui:
                try:
                    # Extract the vision component (first 128 elements)
                    predicted_latent_vector = predicted_next_vector[:128]

                    # Decode the latent vector back into an image
                    predicted_image = self.vae.decode(predicted_latent_vector)

                    # Update the GUI's prediction panel
                    self.gui.update_prediction(predicted_image)

                except Exception as e:
                    # Log any errors during the decoding/display process
                    error_msg = f"Error decoding/displaying prediction: {e}"
                    self.gui.log_system(error_msg)

            # 5. Store the current prediction and vector for the next tick's comparison
            self.prediction_from_previous_tick = predicted_next_vector
            self.previous_sensory_vector = current_sensory_vector
            
            # --- Hash Info Management & State Tracking ---
            
            # Track the change in internal state from the last action
            if self.last_tick_hash and self.state_before_action:
                # Get the ID of the hash that *led to* the last action
                last_hash_id, _, _ = self.hash_db.get_or_create_hash_info(self.last_tick_hash)

                # Calculate the change in state
                state_changes = {
                    driver: self.internal_state[driver] - self.state_before_action[driver]
                    for driver in self.internal_state
                }

                # Update the database with the observed impact
                self.hash_db.update_state_impact(last_hash_id, state_changes)

            # Get or create the info for the CURRENT hash
            current_hash = self.current_sensory_packet.get('lsh_hash', '')
            if current_hash:
                hash_id, hash_info, is_new = self.hash_db.get_or_create_hash_info(current_hash)

                # ONE-TIME CALIBRATION FOR NEW HASHES
                if is_new:
                    if self.gui:
                        self.gui.log_system(f"New hash {hash_id} discovered! Calibrating...")
                    # TODO: Implement the one-time calibration logic here.
                    # This involves creating a temporary neuro_vector and passing it
                    # through the D-LSTM at cascading depths to find the optimal depth.
                    # For now, we'll set placeholder depths.
                    placeholder_depths = {'optimal': 6, 'streamlined': 2, 'deep': 8}
                    self.hash_db.update_depths(hash_id, placeholder_depths)

                # Store the current internal state *before* the next action is taken
                self.state_before_action = self.internal_state.copy()
            
            # --- Main OLM Pipeline ---
            if 'vision' in self.current_sensory_packet or 'text' in self.current_sensory_packet or 'keyboard' in self.current_sensory_packet:
                # 1. P-LSTM: Discover patterns from the sensory vector
                pattern_vector = self.pattern_recognizer.process(current_sensory_vector)

                # 2. C-LSTM: Compress patterns into a neurotransmitter vector
                neuro_vector = self.neuro_activator.process(pattern_vector)

                # 3. D-LSTMs: Generate internal thought and external text output
                
                # Get the optimal processing depth for the current state
                optimal_depth = None
                if current_hash and 'hash_info' in locals():
                    optimal_depth = hash_info.get('optimal_depth')

                # 3a. Process internal thought
                thought_vector = self.thought_d_lstm.process(neuro_vector, self.internal_state, depth=optimal_depth)
                # Convert vector to integer tokens and then to text
                thought_tokens = np.round(thought_vector).astype(int)
                thought_text = self.tokenizer.detokenize(thought_tokens)

                # 3b. Process external text output
                text_vector = self.text_d_lstm.process(neuro_vector, self.internal_state, depth=optimal_depth)
                # Convert vector to integer tokens and then to text
                output_tokens = np.round(text_vector).astype(int)
                output_text = self.tokenizer.detokenize(output_tokens)

                # 4. Log and route the outputs
                if self.gui:
                    self.gui.log_thoughts(thought_text)
                    self.gui.log_speech(output_text)

                # Feed the external output back into the system for the next tick
                self.text_handler.add_message(output_text, source='olm')

                # TODO: Pass the external output to the TinyLlama model
            
            # Periodically save tokenizer state to persist vocabulary growth
            if tick_count % 200 == 0:
                self.tokenizer.save_state()
                # Also save the prediction model state
                predictor_checkpoint_file = os.path.join(self.checkpoints_dir, "prediction_model.pth")
                self.predictor.save_model(predictor_checkpoint_file)
                if self.gui:
                    self.gui.log_system("Periodically saved prediction model.")

            # Log sensory packet summary every 20 ticks
            if tick_count % 20 == 0 and self.gui:
                summary_parts = []
                
                if 'vision' in self.current_sensory_packet:
                    summary_parts.append("Vision: Yes")
                
                if 'text' in self.current_sensory_packet:
                    # Handle tokenized text for display
                    text_data = self.current_sensory_packet['text']['text']
                    if isinstance(text_data, list):  # Tokenized text
                        text_preview = f"[{len(text_data)} tokens]"
                    else:  # Raw text (fallback)
                        text_preview = str(text_data)[:20]
                        if len(str(text_data)) > 20:
                            text_preview += "..."
                    summary_parts.append(f"Text: {text_preview}")
                
                if 'keyboard' in self.current_sensory_packet:
                    event_count = len(self.current_sensory_packet['keyboard']['events'])
                    summary_parts.append(f"Keyboard: {event_count} events")
                
                if 'lsh_hash' in self.current_sensory_packet:
                    hash_preview = self.current_sensory_packet['lsh_hash'][:8]
                    summary_parts.append(f"Hash: {hash_preview}...")
                
                if summary_parts:
                    summary_string = f"Packet: {' | '.join(summary_parts)}"
                    self.gui.log_sensory_packet(summary_string)
                else:
                    self.gui.log_sensory_packet("Packet: Empty")
            
            # TODO: Pass the final sensory_packet to the OLM's main tick method
            
            # --- TICK LOGIC GOES HERE ---
            tick_count += 1
            tick_message = f"Tick! #{tick_count}"
            print(tick_message)
            
            # Update TPS counter in GUI
            if self.gui:
                self.gui.update_tps(tick_count)
            
            # Also log to console log if GUI is available
            if self.gui:
                self.gui.log_console(tick_message)
            
            # Calculate time elapsed for this tick's logic
            elapsed_time = time.time() - start_time
            
            # Calculate required sleep time to maintain tick rate
            sleep_time = (1.0 / self.tick_rate) - elapsed_time
            
            # Sleep if there's time remaining in the tick
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Log if we're falling behind
                warning_message = f"Warning: Tick #{tick_count} took {elapsed_time:.3f}s (target: {1.0/self.tick_rate:.3f}s)"
                if self.gui:
                    self.gui.log_system(warning_message)
                else:
                    print(warning_message)
                    
        # Engine has stopped
        stop_message = "Engine stopped."
        if self.gui:
            self.gui.log_system(stop_message)
        else:
            print(stop_message)
            
    def ensure_checkpoints_dir(self):
        """Create checkpoints directory if it doesn't exist"""
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
            if self.gui:
                self.gui.log_system(f"Created checkpoints directory: {self.checkpoints_dir}")
            else:
                print(f"Created checkpoints directory: {self.checkpoints_dir}")
                
    def wipe_checkpoints(self):
        """Delete all files in the checkpoints directory"""
        if os.path.exists(self.checkpoints_dir):
            try:
                # Get list of all files in the directory
                files = os.listdir(self.checkpoints_dir)
                deleted_count = 0
                
                for file in files:
                    file_path = os.path.join(self.checkpoints_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                        deleted_count += 1
                
                if self.gui:
                    self.gui.log_system(f"Wiped checkpoints directory: {deleted_count} items deleted")
                else:
                    print(f"Wiped checkpoints directory: {deleted_count} items deleted")
                    
            except Exception as e:
                error_msg = f"Error wiping checkpoints: {str(e)}"
                if self.gui:
                    self.gui.log_system(error_msg)
                else:
                    print(error_msg)
        else:
            if self.gui:
                self.gui.log_system("Checkpoints directory does not exist")
            else:
                print("Checkpoints directory does not exist")
