import pynput.keyboard
import threading
import time
import queue

# Configuration constants
HOLD_THRESHOLD = 0.2  # Minimum duration (seconds) to consider a key as "held"

class KeyboardListener:
    def __init__(self, pause_threshold=1.0):
        """
        Initialize the keyboard listener
        
        Args:
            pause_threshold (float): Time in seconds to wait after last activity before packaging an utterance
        """
        self.pause_threshold = pause_threshold
        
        # Thread-safe queue for sending completed utterances to the engine
        self.message_queue = queue.Queue()
        
        # Buffer to store keystrokes and hold events
        self.event_buffer = []
        
        # Dictionary to track currently held keys and their press times
        self.held_keys = {}
        
        # Track the last time any keyboard activity occurred
        self.last_activity_time = time.time()
        
        # Event for clean shutdown
        self.stop_event = threading.Event()
        
        # Threading components
        self.listener = None
        self.monitor_thread = None
    
    def _on_press(self, key):
        """Callback for key press events"""
        try:
            # Get the character or name of the key
            if hasattr(key, 'char') and key.char is not None:
                key_str = key.char
            else:
                key_str = str(key)
            
            # Check if this key is already being held (auto-repeat from OS)
            if key_str in self.held_keys:
                return  # Ignore auto-repeat events
            
            # Record the current time for this key press
            current_time = time.time()
            self.held_keys[key_str] = current_time
            
            # Calculate time since last event before updating activity time
            time_since_last = current_time - self.last_activity_time if self.event_buffer else 0
            
            # Update last activity time
            self.last_activity_time = current_time
            
            # Add press event to buffer
            self.event_buffer.append({
                'type': 'press',
                'key': key_str,
                'time_since_last': time_since_last,
                'timestamp': current_time
            })
            
        except Exception as e:
            print(f"Error in _on_press: {e}")
    
    def _on_release(self, key):
        """Callback for key release events"""
        try:
            # Get the character or name of the key
            if hasattr(key, 'char') and key.char is not None:
                key_str = key.char
            else:
                key_str = str(key)
            
            # Check if this key was being held
            if key_str in self.held_keys:
                # Calculate hold duration
                press_time = self.held_keys[key_str]
                current_time = time.time()
                hold_duration = current_time - press_time
                
                # If hold duration is significant, add a hold event
                if hold_duration > HOLD_THRESHOLD:
                    self.event_buffer.append({
                        'type': 'hold',
                        'key': key_str,
                        'hold_duration': hold_duration,
                        'timestamp': current_time
                    })
                
                # Remove the key from held keys
                del self.held_keys[key_str]
                
                # Update last activity time
                self.last_activity_time = current_time
                
        except Exception as e:
            print(f"Error in _on_release: {e}")
    
    def _activity_monitor(self):
        """Monitor activity and package utterances when pauses are detected"""
        while not self.stop_event.is_set():
            current_time = time.time()
            time_since_activity = current_time - self.last_activity_time
            
            # Check if we should package an utterance
            # Conditions: buffer not empty, no keys held, pause threshold exceeded
            if (self.event_buffer and 
                not self.held_keys and 
                time_since_activity > self.pause_threshold):
                
                # Package the utterance
                utterance = {
                    'type': 'keyboard_utterance',
                    'events': self.event_buffer.copy(),
                    'total_duration': time_since_activity,
                    'timestamp': current_time
                }
                
                # Send to message queue
                try:
                    self.message_queue.put(utterance, timeout=0.1)
                except queue.Full:
                    print("Warning: Keyboard message queue is full")
                
                # Clear the buffer
                self.event_buffer.clear()
            
            # Sleep for a short duration
            time.sleep(0.1)
    
    def start(self):
        """Start the keyboard listener and activity monitor"""
        if self.listener is None:
            # Start the keyboard listener
            self.listener = pynput.keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self.listener.start()
            
            # Start the activity monitor thread
            self.monitor_thread = threading.Thread(target=self._activity_monitor, daemon=True)
            self.monitor_thread.start()
            
            print("Keyboard listener started")
    
    def stop(self):
        """Stop the keyboard listener and clean up"""
        # Signal stop event
        self.stop_event.set()
        
        # Stop the keyboard listener
        if self.listener:
            self.listener.stop()
            self.listener = None
        
        # Wait for monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        print("Keyboard listener stopped")
    
    def get_latest_utterance(self):
        """
        Get the latest utterance from the message queue (non-blocking)
        
        Returns:
            dict or None: The utterance dictionary if available, None otherwise
        """
        try:
            return self.message_queue.get_nowait()
        except queue.Empty:
            return None

if __name__ == "__main__":
    # Demonstration usage
    print("Creating KeyboardListener...")
    listener = KeyboardListener(pause_threshold=1.0)
    
    print("Starting keyboard listener...")
    print("Type some text and then pause for 1 second to see utterances packaged.")
    print("Press Ctrl+C to stop.")
    
    try:
        listener.start()
        
        # Monitor for utterances
        utterance_count = 0
        while True:
            utterance = listener.get_latest_utterance()
            if utterance:
                utterance_count += 1
                print(f"\n--- Utterance #{utterance_count} ---")
                print(f"Total duration: {utterance['total_duration']:.2f}s")
                print(f"Number of events: {len(utterance['events'])}")
                
                for i, event in enumerate(utterance['events']):
                    if event['type'] == 'press':
                        print(f"  {i+1}. Press: '{event['key']}' (after {event['time_since_last']:.2f}s)")
                    elif event['type'] == 'hold':
                        print(f"  {i+1}. Hold: '{event['key']}' for {event['hold_duration']:.2f}s")
                
                print("--- End Utterance ---\n")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping keyboard listener...")
        listener.stop()
        print("Demonstration completed!")
