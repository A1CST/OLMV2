import threading
import queue
import requests
import time


class TinyLlamaManager:
    def __init__(self, engine_text_handler, base_url="http://localhost:5000"):
        """
        Initialize the TinyLlama manager.
        Args:
            engine_text_handler: A reference to the engine's text_handler.
            base_url (str): The base URL of the local model server.
        """
        self.text_handler = engine_text_handler
        self.endpoint = f"{base_url}/v1/chat/completions"
        self.message_history = []
        self.system_prompt = (
            "Just talk to the user like a normal person. Responses should be less than 30 words. "
            "If the user mentions reading, dreams, or learning, engage with those topics enthusiastically. "
            "Ask follow-up questions to keep the conversation flowing."
        )
        
        self.request_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _process_queue(self):
        """The worker thread that processes requests from the queue."""
        while True:
            input_text = self.request_queue.get()
            if input_text is None:  # Shutdown signal
                break
            
            # --- REAL MODEL INFERENCE ---
            print(f"[TinyLlama] Sending to model: '{input_text}'")
            try:
                messages = [{"role": "system", "content": self.system_prompt}]
                # Include the last 7 exchanges for context
                for exchange in self.message_history[-7:]:
                    messages.append({"role": "user", "content": exchange["user"]})
                    messages.append({"role": "assistant", "content": exchange["assistant"]})
                messages.append({"role": "user", "content": input_text})
                
                payload = {
                    "model": "tinyllama",
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.8,
                }
                
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    assistant_response = data['choices'][0]['message']['content'].strip()
                    
                    self.message_history.append({
                        "user": input_text,
                        "assistant": assistant_response,
                    })
                    # Keep last 10 exchanges in history
                    if len(self.message_history) > 10:
                        self.message_history.pop(0)
                        
                    response_text = assistant_response
                else:
                    response_text = f"Error: HTTP {response.status_code}"
            
            except requests.exceptions.RequestException as e:
                response_text = f"Connection error: {e}"
            
            print(f"[TinyLlama] Generated response: '{response_text}'")
            # Add the response back to the engine's input queue
            self.text_handler.add_message(response_text, source='tinyllama')

    def submit_prompt(self, text):
        """Submits a prompt to be processed by the TinyLlama model."""
        self.request_queue.put(text)

    def stop(self):
        """Stops the worker thread."""
        self.request_queue.put(None)


