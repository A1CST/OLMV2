class TextInputHandler:
    def __init__(self):
        """Initialize the text input handler with an empty message queue"""
        self.message_queue = []
    
    def add_message(self, text, source):
        """
        Add a text message to the queue
        
        Args:
            text (str): The text message content
            source (str): The source of the message (e.g., 'user', 'tinyllama')
        """
        message = {
            'source': source,
            'text': text
        }
        self.message_queue.append(message)
    
    def get_latest_input(self):
        """
        Retrieve the oldest message from the queue
        
        Returns:
            dict or None: The message dictionary with 'source' and 'text' keys, or None if queue is empty
        """
        if self.message_queue:
            return self.message_queue.pop(0)  # Remove and return the oldest message
        else:
            return None

if __name__ == "__main__":
    # Demonstration usage
    print("Creating TextInputHandler...")
    handler = TextInputHandler()
    
    # Add some test messages
    print("\nAdding messages to the queue...")
    handler.add_message("Hello, how are you?", "user")
    handler.add_message("I'm doing well, thank you!", "tinyllama")
    handler.add_message("What's the weather like?", "user")
    handler.add_message("It's sunny and 75 degrees.", "tinyllama")
    
    print(f"Queue length after adding messages: {len(handler.message_queue)}")
    
    # Retrieve and print all messages
    print("\nRetrieving messages from the queue:")
    message_count = 0
    while True:
        message = handler.get_latest_input()
        if message is None:
            break
        
        message_count += 1
        print(f"Message {message_count}: [{message['source']}] '{message['text']}'")
    
    print(f"\nTotal messages processed: {message_count}")
    print(f"Queue length after processing: {len(handler.message_queue)}")
    print("TextInputHandler demonstration completed successfully!")
