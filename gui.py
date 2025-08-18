import tkinter as tk
from tkinter import ttk
import sys
import io
from PIL import Image, ImageTk

class App:
    def __init__(self, engine=None):
        # Store engine reference
        self.engine = engine
        
        # TPS tracking variables
        self.tps_start_time = None
        self.tps_tick_count = 0
        self.tps_history = []  # Store recent TPS values for averaging
        self.target_tps = 20.0
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("OLM Observer")
        self.root.configure(bg='#2b2b2b')  # Dark grey background
        
        # Set window size and make it resizable
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        
        # Set up close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_top_section()
        self.setup_right_sidebar()
        self.setup_bottom_section()
        
        # Set up console output redirection
        self.setup_console_redirection()
        
    def setup_top_section(self):
        """Create the top section with vision displays"""
        # Main container for top section
        top_frame = tk.Frame(self.root, bg='#2b2b2b')
        top_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Configure grid weights for top frame
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=1)
        
        # Left side - Current Vision
        vision_frame = tk.Frame(top_frame, bg='#2b2b2b')
        vision_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # Create a canvas for vision display instead of a label
        vision_canvas = tk.Canvas(vision_frame, 
                                 bg='#1e1e1e', 
                                 relief='solid', bd=1,
                                 width=600, height=400)  # Set actual pixel dimensions
        vision_canvas.pack(pady=10, fill='both', expand=True)
        
        # Add a label on top of the canvas
        vision_canvas.create_text(300, 20, text="Vision Input", 
                                 fill='white', font=('Arial', 12, 'bold'))
        
        # Right side - Predicted Vision
        prediction_frame = tk.Frame(top_frame, bg='#2b2b2b')
        prediction_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        # Create a canvas for prediction display
        prediction_canvas = tk.Canvas(prediction_frame, 
                                     bg='#1e1e1e', 
                                     relief='solid', bd=1,
                                     width=600, height=400)  # Set actual pixel dimensions
        prediction_canvas.pack(pady=10, fill='both', expand=True)
        
        # Add a label on top of the canvas
        prediction_canvas.create_text(300, 20, text="Prediction Output", 
                                     fill='white', font=('Arial', 12, 'bold'))
        
        # Store references for later updates
        self.vision_canvas = vision_canvas
        self.prediction_canvas = prediction_canvas
        
    def setup_right_sidebar(self):
        """Create the right sidebar for internal state"""
        # Right sidebar container
        sidebar_frame = tk.Frame(self.root, bg='#2b2b2b', width=300)
        sidebar_frame.grid(row=0, column=1, sticky='nsew', padx=(0, 10), pady=10)
        sidebar_frame.grid_propagate(False)  # Maintain fixed width
        
        # Title
        title_label = tk.Label(sidebar_frame, text="Internal State", 
                              bg='#2b2b2b', fg='white', 
                              font=('Arial', 14, 'bold'))
        title_label.pack(pady=(10, 5))
        
        # TPS counter frame
        tps_frame = tk.Frame(sidebar_frame, bg='#2b2b2b')
        tps_frame.pack(fill='x', padx=10, pady=(5, 5))
        
        # TPS label
        tps_label = tk.Label(tps_frame, text="TPS Counter", 
                           bg='#2b2b2b', fg='white', 
                           font=('Arial', 10, 'bold'))
        tps_label.pack(anchor='w', pady=(0, 5))
        
        # TPS display
        self.tps_label = tk.Label(tps_frame, 
                                text="Current TPS: 0.0\n"
                                     "Target TPS: 20.0\n"
                                     "Average TPS: 0.0",
                                bg='#1e1e1e', fg='#00FF00',  # Green text for TPS
                                font=('Consolas', 10, 'bold'),
                                justify=tk.LEFT, anchor='nw',
                                relief='solid', bd=1)
        self.tps_label.pack(fill='x', pady=(0, 5))
        
        # Prediction Error frame
        error_frame = tk.Frame(sidebar_frame, bg='#2b2b2b')
        error_frame.pack(fill='x', padx=10, pady=(5, 5))
        
        # Prediction Error label
        error_label = tk.Label(error_frame, text="Prediction Error", 
                             bg='#2b2b2b', fg='white', 
                             font=('Arial', 10, 'bold'))
        error_label.pack(anchor='w', pady=(0, 5))
        
        # Prediction Error display
        self.error_label = tk.Label(error_frame, 
                                  text="Error: 0.000000\n"
                                       "Status: No Data",
                                  bg='#1e1e1e', fg='#FFD700',  # Gold text for error
                                  font=('Consolas', 10, 'bold'),
                                  justify=tk.LEFT, anchor='nw',
                                  relief='solid', bd=1)
        self.error_label.pack(fill='x', pady=(0, 5))
        
        # Status display
        self.status_label = tk.Label(sidebar_frame, 
                                    text="State: Awake\n"
                                         "---Constraints---\n"
                                         "Energy: 100.0\n"
                                         "Comfort: 75.0\n"
                                         "Confidence: 50.0\n"
                                         "---Drivers---\n"
                                         "Novelty: 0.0\n"
                                         "Boredom: 0.0",
                                    bg='#1e1e1e', fg='white',
                                    font=('Consolas', 10),
                                    justify=tk.LEFT, anchor='nw',
                                    relief='solid', bd=1)
        self.status_label.pack(fill='both', expand=True, padx=10, pady=5)
        
        # User input frame
        user_input_frame = tk.Frame(sidebar_frame, bg='#2b2b2b')
        user_input_frame.pack(fill='x', padx=10, pady=(10, 5))
        
        # User input label
        user_input_label = tk.Label(user_input_frame, text="User Message Input", 
                                   bg='#2b2b2b', fg='white', 
                                   font=('Arial', 10, 'bold'))
        user_input_label.pack(anchor='w', pady=(0, 5))
        
        # User input entry and send button frame
        input_controls_frame = tk.Frame(user_input_frame, bg='#2b2b2b')
        input_controls_frame.pack(fill='x')
        
        # User input entry
        self.user_input_entry = tk.Entry(input_controls_frame, 
                                        bg='#1e1e1e', fg='white',
                                        font=('Arial', 10),
                                        relief='solid', bd=1)
        self.user_input_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # Send button
        self.send_button = tk.Button(input_controls_frame, text="Send", 
                                   bg='#2196F3', fg='white',
                                   font=('Arial', 10, 'bold'),
                                   command=self.send_user_message,
                                   relief='flat', bd=0,
                                   width=8)
        self.send_button.pack(side='right')
        
        # Bind Enter key to send message
        self.user_input_entry.bind('<Return>', lambda event: self.send_user_message())
        
        # Control buttons frame
        control_frame = tk.Frame(sidebar_frame, bg='#2b2b2b')
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Start button
        self.start_button = tk.Button(control_frame, text="Start", 
                                     bg='#4CAF50', fg='white',
                                     font=('Arial', 12, 'bold'),
                                     command=self.start_engine,
                                     relief='flat', bd=0,
                                     width=10, height=2)
        self.start_button.pack(side='left', padx=(0, 5))
        
        # Stop button
        self.stop_button = tk.Button(control_frame, text="Stop", 
                                    bg='#f44336', fg='white',
                                    font=('Arial', 12, 'bold'),
                                    command=self.stop_engine,
                                    relief='flat', bd=0,
                                    width=10, height=2)
        self.stop_button.pack(side='left', padx=(5, 5))
        
        # Wipe button
        self.wipe_button = tk.Button(control_frame, text="Wipe", 
                                    bg='#FF9800', fg='white',
                                    font=('Arial', 12, 'bold'),
                                    command=self.wipe_checkpoints,
                                    relief='flat', bd=0,
                                    width=10, height=2)
        self.wipe_button.pack(side='right', padx=(5, 0))
        
    def setup_bottom_section(self):
        """Create the bottom section with five log panels"""
        # Bottom container
        bottom_frame = tk.Frame(self.root, bg='#2b2b2b')
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=(0, 10))
        
        # Configure grid weights for five columns
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, weight=1)
        bottom_frame.grid_columnconfigure(3, weight=1)
        bottom_frame.grid_columnconfigure(4, weight=1)
        
        # Speech Log (left)
        self.setup_log_panel(bottom_frame, "Speech Log", 0, "speech_log")
        
        # System Log (middle-left)
        self.setup_log_panel(bottom_frame, "System Log", 1, "system_log")
        
        # Console Log (middle)
        self.setup_log_panel(bottom_frame, "Console Log", 2, "console_log")
        
        # Internal Thoughts Log (right)
        self.setup_log_panel(bottom_frame, "Internal Thoughts", 3, "thoughts_log")
        
        # Sensory Packet Log (far right)
        self.setup_log_panel(bottom_frame, "Sensory Packet Log", 4, "sensory_log")
        
    def setup_log_panel(self, parent, title, column, attr_name):
        """Helper method to create a log panel"""
        # Container for this log
        log_container = tk.Frame(parent, bg='#2b2b2b')
        # Add padding between panels, but not after the last one
        if column < 4:  # All but the last column
            log_container.grid(row=0, column=column, sticky='nsew', padx=(0, 5))
        else:  # Last column
            log_container.grid(row=0, column=column, sticky='nsew', padx=(0, 0))
        
        # Label
        label = tk.Label(log_container, text=title, 
                        bg='#2b2b2b', fg='white', 
                        font=('Arial', 10, 'bold'))
        label.pack(anchor='w', pady=(0, 5))
        
        # Text widget
        text_widget = tk.Text(log_container, 
                             bg='#1e1e1e', fg='white',
                             font=('Consolas', 9),
                             height=8,  # Reduced height for three panels
                             state='disabled',  # Read-only
                             relief='solid', bd=1)
        text_widget.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(log_container, orient='vertical', command=text_widget.yview)
        scrollbar.pack(side='right', fill='y')
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Store reference
        setattr(self, attr_name, text_widget)
        
    def setup_console_redirection(self):
        """Set up console output redirection to capture print statements"""
        # Store original stdout
        self.original_stdout = sys.stdout
        
        # Create a custom stdout that redirects to our console log
        class ConsoleRedirector:
            def __init__(self, gui_app):
                self.gui_app = gui_app
                self.buffer = ""
                
            def write(self, text):
                # Write to original stdout first
                self.gui_app.original_stdout.write(text)
                
                # Add to buffer
                self.buffer += text
                
                # If we have a complete line, log it
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    # Log all complete lines
                    for line in lines[:-1]:  # All but the last (incomplete) line
                        if line.strip():  # Only log non-empty lines
                            self.gui_app.log_console(line.strip())
                    # Keep the incomplete line in buffer
                    self.buffer = lines[-1]
                    
            def flush(self):
                self.gui_app.original_stdout.flush()
                
        # Set up the redirector
        self.console_redirector = ConsoleRedirector(self)
        sys.stdout = self.console_redirector
        
    def update_vision(self, image):
        """Update the current vision display with a PIL Image"""
        try:
            # Get the actual size of the vision canvas widget
            canvas_width = self.vision_canvas.winfo_width()
            canvas_height = self.vision_canvas.winfo_height()
            
            # If the canvas hasn't been rendered yet, use default sizes
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600  # Larger default width
                canvas_height = 400  # Larger default height
            
            # Calculate aspect ratio to maintain proportions
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height
            display_aspect = canvas_width / canvas_height
            
            # Log the image and display dimensions for debugging
            self.log_system(f"Image size: {img_width}x{img_height}, Display size: {canvas_width}x{canvas_height}")
            
            if aspect_ratio > display_aspect:
                # Image is wider than display area - fit to width
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            else:
                # Image is taller than display area - fit to height
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            
            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for tkinter
            photo_image = ImageTk.PhotoImage(resized_image)
            
            # Clear the canvas and display the new image
            self.vision_canvas.delete("all")
            
            # Add the title text back
            self.vision_canvas.create_text(canvas_width//2, 20, text="Vision Input", 
                                          fill='white', font=('Arial', 12, 'bold'))
            
            # Calculate position to center the image
            x_pos = (canvas_width - new_width) // 2
            y_pos = (canvas_height - new_height) // 2 + 30  # Offset for title
            
            # Display the image on the canvas
            self.vision_canvas.create_image(x_pos, y_pos, anchor='nw', image=photo_image)
            
            # Keep a reference to prevent garbage collection
            self.vision_canvas.image = photo_image
            
        except Exception as e:
            # Fallback to text if image display fails
            self.vision_canvas.delete("all")
            self.vision_canvas.create_text(300, 200, text=f"Vision Error: {str(e)}", 
                                          fill='red', font=('Arial', 12))
            self.log_system(f"Error updating vision display: {str(e)}")
        
    def update_prediction(self, image):
        """Update the predicted vision display with a PIL Image"""
        try:
            # Get the actual size of the prediction canvas widget
            canvas_width = self.prediction_canvas.winfo_width()
            canvas_height = self.prediction_canvas.winfo_height()
            
            # If the canvas hasn't been rendered yet, use default sizes
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600  # Larger default width
                canvas_height = 400  # Larger default height
            
            # Calculate aspect ratio to maintain proportions
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height
            display_aspect = canvas_width / canvas_height
            
            if aspect_ratio > display_aspect:
                # Image is wider than display area - fit to width
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            else:
                # Image is taller than display area - fit to height
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            
            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for tkinter
            photo_image = ImageTk.PhotoImage(resized_image)
            
            # Clear the canvas and display the new image
            self.prediction_canvas.delete("all")
            
            # Add the title text back
            self.prediction_canvas.create_text(canvas_width//2, 20, text="Prediction Output", 
                                              fill='white', font=('Arial', 12, 'bold'))
            
            # Calculate position to center the image
            x_pos = (canvas_width - new_width) // 2
            y_pos = (canvas_height - new_height) // 2 + 30  # Offset for title
            
            # Display the image on the canvas
            self.prediction_canvas.create_image(x_pos, y_pos, anchor='nw', image=photo_image)
            
            # Keep a reference to prevent garbage collection
            self.prediction_canvas.image = photo_image
            
        except Exception as e:
            # Fallback to text if image display fails
            self.prediction_canvas.delete("all")
            self.prediction_canvas.create_text(300, 200, text=f"Prediction Error: {str(e)}", 
                                              fill='red', font=('Arial', 12))
            self.log_system(f"Error updating prediction display: {str(e)}")
        
    def update_status(self, status_text):
        """Update the internal state display"""
        # Placeholder method - will be implemented later
        pass
    
    def update_tps(self, tick_count):
        """Update the TPS counter display"""
        import time
        
        current_time = time.time()
        
        # Initialize start time if not set
        if self.tps_start_time is None:
            self.tps_start_time = current_time
            self.tps_tick_count = 0
        
        # Update tick count
        self.tps_tick_count = tick_count
        
        # Calculate current TPS
        elapsed_time = current_time - self.tps_start_time
        if elapsed_time > 0:
            current_tps = self.tps_tick_count / elapsed_time
        else:
            current_tps = 0.0
        
        # Store in history for averaging (keep last 10 values)
        self.tps_history.append(current_tps)
        if len(self.tps_history) > 10:
            self.tps_history.pop(0)
        
        # Calculate average TPS
        if self.tps_history:
            average_tps = sum(self.tps_history) / len(self.tps_history)
        else:
            average_tps = 0.0
        
        # Update display
        tps_text = f"Current TPS: {current_tps:.1f}\n"
        tps_text += f"Target TPS: {self.target_tps:.1f}\n"
        tps_text += f"Average TPS: {average_tps:.1f}"
        
        # Color coding based on performance
        if current_tps >= self.target_tps * 0.9:  # Within 90% of target
            color = '#00FF00'  # Green
        elif current_tps >= self.target_tps * 0.7:  # Within 70% of target
            color = '#FFFF00'  # Yellow
        else:
            color = '#FF0000'  # Red
        
        self.tps_label.config(text=tps_text, fg=color)
    
    def update_prediction_error(self, error_value):
        """Update the prediction error display"""
        if error_value is None:
            error_text = "Error: 0.000000\nStatus: No Data"
            color = '#FFD700'  # Gold
        else:
            error_text = f"Error: {error_value:.6f}\nStatus: Active"
            
            # Color coding based on error magnitude
            if error_value < 0.001:  # Very low error
                color = '#00FF00'  # Green
            elif error_value < 0.01:  # Low error
                color = '#FFFF00'  # Yellow
            elif error_value < 0.1:  # Medium error
                color = '#FFA500'  # Orange
            else:  # High error
                color = '#FF0000'  # Red
        
        self.error_label.config(text=error_text, fg=color)
    
    def reset_tps_counter(self):
        """Reset the TPS counter"""
        self.tps_start_time = None
        self.tps_tick_count = 0
        self.tps_history.clear()
        self.update_tps(0)
        
    def log_action(self, action_text):
        """Add an action to the speech log (since action log was replaced)"""
        self._add_to_log(self.speech_log, action_text)
        
    def log_system(self, system_text):
        """Add a system message to the system log"""
        self._add_to_log(self.system_log, system_text)
        
    def log_console(self, console_text):
        """Add a message to the console log (mirrors console output)"""
        self._add_to_log(self.console_log, console_text)
        
    def log_sensory_packet(self, packet_summary):
        """Add a sensory packet summary to the sensory packet log"""
        self._add_to_log(self.sensory_log, packet_summary)
        
    def log_thoughts(self, thought_text):
        """Add an internal thought to the thoughts log"""
        self._add_to_log(self.thoughts_log, thought_text)
        
    def log_speech(self, speech_text):
        """Add speech output to the speech log"""
        self._add_to_log(self.speech_log, speech_text)
        

        
    def _add_to_log(self, text_widget, message):
        """Helper method to add a message to any log widget"""
        # Enable the text widget for editing
        text_widget.config(state='normal')
        
        # Add timestamp and message
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Insert at the end
        text_widget.insert(tk.END, log_entry)
        
        # Scroll to the bottom
        text_widget.see(tk.END)
        
        # Disable the text widget again
        text_widget.config(state='disabled')
        
    def start_engine(self):
        """Start the engine"""
        if self.engine:
            self.log_action("Start button clicked - starting engine")
            self.engine.start()
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
        else:
            self.log_system("ERROR: No engine connected!")
        
    def stop_engine(self):
        """Stop the engine"""
        if self.engine:
            self.log_action("Stop button clicked - stopping engine")
            self.engine.stop()
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
        else:
            self.log_system("ERROR: No engine connected!")
            
    def wipe_checkpoints(self):
        """Wipe the checkpoints directory"""
        if self.engine:
            self.log_action("Wipe button clicked - clearing checkpoints")
            self.engine.wipe_checkpoints()
        else:
            self.log_system("ERROR: No engine connected!")
            
    def send_user_message(self):
        """Send a user message to the engine's text handler"""
        if self.engine:
            # Get the text from the entry widget
            text = self.user_input_entry.get().strip()
            
            if text:  # Only send if text is not empty
                # Add the message to the engine's text handler
                self.engine.text_handler.add_message(text, 'user')
                
                # Log the action
                self.log_action(f"Sent user message: {text}")
                
                # Clear the entry widget
                self.user_input_entry.delete(0, tk.END)
            else:
                # Log if user tried to send empty message
                self.log_system("Attempted to send empty message - ignored")
        else:
            self.log_system("ERROR: No engine connected!")
            
    def on_closing(self):
        """Handle window closing"""
        # Restore original stdout
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
            
        self.stop_engine()
        self.root.destroy()
        
    def run(self):
        """Start the GUI application"""
        # Initially disable stop button
        self.stop_button.config(state='disabled')
        self.root.mainloop()

if __name__ == "__main__":
    # Create engine instance
    from engine import Engine
    engine = Engine()
    
    # Create GUI instance with engine reference
    app = App(engine)
    
    # Connect GUI back to engine
    engine.gui = app
    
    # Start the application
    app.run()
