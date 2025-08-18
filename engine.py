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
from prediction_model import StatePredictor, MAX_TEXT_TOKENS, PREVIOUS_SPEECH_SIZE
from tinyllama_integration import TinyLlamaManager
from P_C_pipe import PatternRecognizer, NeurotransmitterActivator
from D_pipe import ThoughtD_LSTM, TextD_LSTM
from hash_info_db import HashInfoDB

# Configuration constants
VISION_CAPTURE_INTERVAL = 10  # Capture screen every N ticks
CHECKPOINT_SAVE_INTERVAL = 200  # Save checkpoints every N ticks
PACKET_SUMMARY_INTERVAL = 20  # Log sensory packet summary every N ticks
STABILIZATION_THRESHOLD = 0.1  # Threshold for calibration stabilization (10%)

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
        
        # Sleep state management
        self.is_sleeping = False
        self.sleep_ticks_remaining = 0
        
        # Track OLM's last spoken tokens for sensory awareness
        self.last_olm_speech_tokens = []
        
        # Initialize the TinyLlama integration
        self.tinyllama_manager = TinyLlamaManager(self.text_handler)
        
        # Track the state before an action is taken
        self.state_before_action = None
        
    def _create_sensory_vector(self, sensory_packet):
        """Convert sensory packet dictionary into a fixed-length, NORMALIZED sensory vector"""
        # Initialize zero vectors for each sensory modality
        vision_vector = np.zeros(128)
        text_vector = np.zeros(MAX_TEXT_TOKENS)
        keyboard_vector = np.zeros(3)
        previous_speech_vector = np.zeros(PREVIOUS_SPEECH_SIZE)  # From prediction_model.py

        # --- Process and Normalize Vision Data ---
        if 'vision' in sensory_packet and 'latent_vector' in sensory_packet['vision']:
            raw_vision = sensory_packet['vision']['latent_vector']
            # Normalize vision vector (L2 norm) to a unit vector
            norm = np.linalg.norm(raw_vision)
            if norm > 0:
                vision_vector = raw_vision / norm

        # --- Process and Normalize Text Data ---
        if 'text' in sensory_packet:
            tokens = sensory_packet['text']['text']
            if isinstance(tokens, list):
                padded_tokens = np.zeros(MAX_TEXT_TOKENS)
                limit = min(len(tokens), MAX_TEXT_TOKENS)
                padded_tokens[:limit] = tokens[:limit]
                # Normalize token IDs by dividing by vocabulary size (scales to 0-1 range)
                vocab_size = self.tokenizer.get_vocabulary_size()
                if vocab_size > 1:
                    text_vector = padded_tokens / vocab_size

        # --- Process and Normalize Keyboard Data ---
        if 'keyboard' in sensory_packet:
            keyboard_data = sensory_packet['keyboard']
            events = keyboard_data.get('events', [])
            num_events = len(events)
            total_hold_duration = sum(e.get('hold_duration', 0.0) for e in events if e.get('type') == 'hold')
            total_utterance_duration = keyboard_data.get('total_duration', 0.0)
            
            # Normalize keyboard data by dividing by plausible maximums
            keyboard_vector[0] = min(num_events / 50.0, 1.0)                  # Cap at 50 events
            keyboard_vector[1] = min(total_hold_duration / 5.0, 1.0)          # Cap at 5s hold
            keyboard_vector[2] = min(total_utterance_duration / 10.0, 1.0)    # Cap at 10s utterance

        # --- Process and Normalize Previous Speech Data ---
        if self.last_olm_speech_tokens:
            padded_speech_tokens = np.zeros(PREVIOUS_SPEECH_SIZE)
            limit = min(len(self.last_olm_speech_tokens), PREVIOUS_SPEECH_SIZE)
            padded_speech_tokens[:limit] = self.last_olm_speech_tokens[:limit]
            vocab_size = self.tokenizer.get_vocabulary_size()
            if vocab_size > 1:
                previous_speech_vector = padded_speech_tokens / vocab_size

        # Concatenate all NORMALIZED vectors into the final sensory vector
        sensory_vector = np.concatenate([vision_vector, text_vector, keyboard_vector, previous_speech_vector])
        
        return sensory_vector
        
    def _calibrate_new_hash(self, neuro_vector):
        """
        Performs a one-time calibration for a new hash to find optimal depths.
        
        Args:
            neuro_vector (numpy.ndarray): Neurotransmitter vector to calibrate with
            
        Returns:
            dict: Calibrated depths dictionary with 'optimal', 'streamlined', 'deep' keys
        """
        # Validate input
        if neuro_vector is None or len(neuro_vector) == 0:
            if self.gui:
                self.gui.log_system("Error: Invalid neuro_vector for calibration")
            return {'optimal': 4, 'streamlined': 2, 'deep': 8}  # Safe defaults
        if self.gui:
            self.gui.log_system("--- Starting D-LSTM Calibration ---")

        output_vectors = []
        max_depth = self.text_d_lstm.get_max_depth()

        for depth in range(1, max_depth + 1):
            # We only need to calibrate one of the D-LSTMs as they share an architecture
            vector = self.text_d_lstm.process(neuro_vector, self.internal_state, depth=depth)
            output_vectors.append(vector)
            if self.gui:
                self.gui.log_system(f"  Depth {depth}/{max_depth}: Output Mean = {vector.mean():.6f}")

        # Find where the output stabilizes
        # Since layers have different output sizes, we'll use a simpler heuristic
        # Look for the point where the output magnitude stabilizes
        optimal_depth = max_depth
        if len(output_vectors) > 1:
            # Calculate the magnitude of each output vector
            magnitudes = [np.linalg.norm(vec) for vec in output_vectors]
            
            # Find where the magnitude change becomes small
            stabilization_threshold = STABILIZATION_THRESHOLD
            for i in range(1, len(magnitudes)):
                if magnitudes[i] > 0 and magnitudes[i-1] > 0:  # Prevent division by zero
                    change_ratio = abs(magnitudes[i] - magnitudes[i-1]) / magnitudes[i-1]
                    if change_ratio < stabilization_threshold:
                        optimal_depth = i + 1  # Use the depth that produced the stable magnitude
                        break

        # For now, streamlined and deep are placeholders based on optimal
        streamlined_depth = max(1, optimal_depth // 2)

        calibrated_depths = {
            'optimal': optimal_depth,
            'streamlined': streamlined_depth,
            'deep': max_depth
        }

        if self.gui:
            self.gui.log_system(f"--- Calibration Complete ---")
            self.gui.log_system(f"  > Optimal Depth Found: {optimal_depth}")

        return calibrated_depths
        
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
            
            # Stop the TinyLlama manager
            try:
                self.tinyllama_manager.stop()
            except Exception:
                pass
            
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
            start_time = time.time()
            # --- Sleep/Wake State Management ---
            if self.is_sleeping:
                # --- SLEEP CYCLE LOGIC ---
                if self.gui:
                    self.gui.log_system(f"Sleeping... {self.sleep_ticks_remaining} ticks remaining.")
                    self.gui.update_internal_state(self.internal_state, self.is_sleeping)

                # TODO: Implement bulk training via replay and hallucination here
                # Example: self.predictor.train_on_replayed_experience()

                # Generate internal thoughts at a higher temperature for exploration
                random_neuro_vector = np.random.randn(64).astype(np.float32)
                thought_vector = self.thought_d_lstm.process(random_neuro_vector, self.internal_state, temperature=1.5)
                thought_tokens = np.round(thought_vector).astype(int)
                thought_text = self.tokenizer.detokenize(thought_tokens)
                if self.gui:
                    self.gui.log_thoughts(f"(Dream) {thought_text}")

                self.sleep_ticks_remaining -= 1
                if self.sleep_ticks_remaining <= 0:
                    self.is_sleeping = False
                    self.internal_state['Energy'] = 100.0  # Replenish energy
                    if self.gui:
                        self.gui.log_system("Waking up. Energy replenished.")
                        self.gui.update_internal_state(self.internal_state, self.is_sleeping)

                # Sleep for the standard tick duration and skip the wake cycle work
                time.sleep(1.0 / self.tick_rate)
                continue

            # --- WAKE CYCLE LOGIC ---
            
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
            
            # Check if we should capture screen
            vision_data = None
            if tick_count % VISION_CAPTURE_INTERVAL == 0:
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

                # Placeholder for depths, will be calibrated in the main pipeline if new
                optimal_depth = hash_info.get('optimal_depth')

                # Store the current internal state *before* the next action is taken
                self.state_before_action = self.internal_state.copy()
            
                        # --- Main OLM Pipeline ---
            if 'vision' in self.current_sensory_packet or 'text' in self.current_sensory_packet or 'keyboard' in self.current_sensory_packet:
                # 1. P-LSTM: Discover patterns from the sensory vector
                pattern_vector = self.pattern_recognizer.process(current_sensory_vector)

                # 2. C-LSTM: Compress patterns into a neurotransmitter vector
                neuro_vector = self.neuro_activator.process(pattern_vector)

                # >> C-LSTM Training Step <<
                # Train the Neuro Activator to better represent the current pattern vector.
                c_loss = self.neuro_activator.train_with_input(pattern_vector, neuro_vector)
                if self.gui and c_loss is not None:
                    self.gui.log_system(f"C-LSTM Training... Loss: {c_loss:.6f}")

                # 3. Check if calibration is needed for the current hash
                if is_new:
                    # Calibrate to find the best depths and save them
                    calibrated_depths = self._calibrate_new_hash(neuro_vector)
                    self.hash_db.update_depths(hash_id, calibrated_depths)
                    optimal_depth = calibrated_depths['optimal']

                # Log which depth is being used for this tick
                if self.gui:
                    log_msg = f"Using stored depth: {optimal_depth}" if not is_new else f"Using newly calibrated depth: {optimal_depth}"
                    self.gui.log_system(log_msg)

                # 4. D-LSTMs: Generate thought and text using the determined depth
                thought_vector = self.thought_d_lstm.process(neuro_vector, self.internal_state, depth=optimal_depth)
                text_vector = self.text_d_lstm.process(neuro_vector, self.internal_state, depth=optimal_depth)

                # >> D-LSTMs Training Step <<
                # Train the D-LSTMs to better reflect the current neuro_vector.
                # This is a simple starting point for training.
                t_loss = self.thought_d_lstm.train(neuro_vector, neuro_vector, optimal_depth)
                s_loss = self.text_d_lstm.train(neuro_vector, neuro_vector, optimal_depth)
                if self.gui and t_loss is not None:
                    self.gui.log_system(f"D-LSTM Training... Thought Loss: {t_loss:.6f}, Speech Loss: {s_loss:.6f}")

                # 5. Token handling and routing
                thought_tokens = np.round(thought_vector).astype(int)
                thought_text = self.tokenizer.detokenize(thought_tokens)

                output_tokens = np.round(text_vector).astype(int)

                # Filter out <unk> tokens (0) for speech
                filtered_output_tokens = [int(tok) for tok in output_tokens if int(tok) != 0]

                # Store original unfiltered speech tokens for next tick sensory vector
                self.last_olm_speech_tokens = output_tokens.tolist()

                if not filtered_output_tokens:
                    output_text = ""
                    if self.gui:
                        self.gui.log_speech("(Silence)")
                        self.gui.log_olm_dialog("(Silence)")
                else:
                    output_text = self.tokenizer.detokenize(filtered_output_tokens)
                    if self.gui:
                        self.gui.log_speech(output_text)
                        self.gui.log_olm_dialog(output_text)

                # Log thoughts
                if self.gui:
                    self.gui.log_thoughts(thought_text)

                # Feed back valid speech and send to TinyLlama
                if output_text:
                    self.text_handler.add_message(output_text, source='olm')
                    try:
                        self.tinyllama_manager.submit_prompt(output_text)
                    except Exception as e:
                        if self.gui:
                            self.gui.log_system(f"TinyLlama submit failed: {e}")

                # Update internal state panel after energy update
                if self.gui:
                    self.gui.update_internal_state(self.internal_state, self.is_sleeping)

                # 6. Calculate and apply energy consumption for this tick
                thought_cost = 0.002
                speech_cost = 0.2 * (optimal_depth if optimal_depth is not None else self.text_d_lstm.get_max_depth())
                decay_cost = 0.15
                total_cost = thought_cost + speech_cost + decay_cost
                self.internal_state['Energy'] -= total_cost

                # 7. Check for state transition to sleep
                if self.internal_state['Energy'] <= 0:
                    self.internal_state['Energy'] = 0
                    self.is_sleeping = True
                    self.sleep_ticks_remaining = 30  # Sleep for 30 ticks
                    if self.gui:
                        self.gui.log_system("Energy depleted. Entering sleep state.")
            
            # Periodically save tokenizer state to persist vocabulary growth
            if tick_count % CHECKPOINT_SAVE_INTERVAL == 0:
                self.tokenizer.save_state()
                # Also save the prediction model state
                predictor_checkpoint_file = os.path.join(self.checkpoints_dir, "prediction_model.pth")
                self.predictor.save_model(predictor_checkpoint_file)
                if self.gui:
                    self.gui.log_system("Periodically saved prediction model.")

            # Log sensory packet summary periodically
            if tick_count % PACKET_SUMMARY_INTERVAL == 0 and self.gui:
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
                # Also keep internal panel refreshed during wake cycle
                self.gui.update_internal_state(self.internal_state, self.is_sleeping)
            
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
