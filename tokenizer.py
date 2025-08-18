import os
import pickle


class CustomTokenizer:
    def __init__(self, checkpoint_path=None):
        """Initialize the custom tokenizer with empty vocabulary"""
        # Dictionary to map words to integer tokens
        self.word_to_token = {}
        
        # Dictionary to map integer tokens back to words
        self.token_to_word = {}
        
        # Counter for the next available token ID
        self.next_token = 0
        
        # Optional path to persist vocabulary
        self.checkpoint_path = checkpoint_path
        
        # Add special token for unknown words
        self.add_word('<unk>')
        
        # Attempt to load previously saved vocabulary
        if self.checkpoint_path:
            self.load_state()
    
    def add_word(self, word):
        """
        Add a word to the vocabulary if it's not already present
        
        Args:
            word (str): The word to add to the vocabulary
            
        Returns:
            int: The token ID for the word
        """
        # Check if the word is already in the vocabulary
        if word not in self.word_to_token:
            # Assign the next available token ID to the word
            self.word_to_token[word] = self.next_token
            self.token_to_word[self.next_token] = word
            
            # Increment the token counter
            self.next_token += 1
        
        return self.word_to_token[word]
    
    def tokenize(self, text):
        """
        Convert a text string into a list of integer tokens
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            list: List of integer tokens representing the text
        """
        # Split the text into words based on spaces
        words = text.split()
        
        # Convert each word to its token ID
        tokens = []
        for word in words:
            # Add the word to vocabulary if it's new and get its token
            token = self.add_word(word)
            tokens.append(token)
        
        return tokens
    
    def detokenize(self, tokens):
        """
        Convert a list of integer tokens back into a text string
        
        Args:
            tokens (list): List of integer tokens to detokenize
            
        Returns:
            str: The reconstructed text string
        """
        words = []
        
        for token in tokens:
            # Look up the word for this token
            if token in self.token_to_word:
                word = self.token_to_word[token]
            else:
                # Use unknown token as fallback
                word = self.token_to_word.get(0, '<unk>')  # Default to first token if available
            
            words.append(word)
        
        # Join the words back into a single string
        return ' '.join(words)
    
    def get_vocabulary_size(self):
        """
        Get the current size of the vocabulary
        
        Returns:
            int: Number of unique words in the vocabulary
        """
        return len(self.word_to_token)
    
    def get_vocabulary(self):
        """
        Get the current vocabulary mapping
        
        Returns:
            dict: Dictionary mapping words to token IDs
        """
        return self.word_to_token.copy()

    def save_state(self):
        """Persist the tokenizer vocabulary to disk if a checkpoint path was provided."""
        if not self.checkpoint_path:
            return
        
        state = {
            'word_to_token': self.word_to_token,
            'token_to_word': self.token_to_word,
            'next_token': self.next_token,
        }
        directory = os.path.dirname(self.checkpoint_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
        except (OSError, pickle.PickleError) as e:
            print(f"Error saving tokenizer state to {self.checkpoint_path}: {e}")

    def load_state(self):
        """Load the tokenizer vocabulary from disk if available and valid."""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return
        try:
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            self.word_to_token = dict(state.get('word_to_token', {}))
            self.token_to_word = dict(state.get('token_to_word', {}))
            self.next_token = int(state.get('next_token', len(self.word_to_token)))
            # Ensure special token exists
            if '<unk>' not in self.word_to_token:
                self.add_word('<unk>')
        except (OSError, pickle.PickleError, ValueError) as e:
            print(f"Warning: could not load tokenizer state from {self.checkpoint_path}: {e}")
            # Reset to minimal valid state
            self.word_to_token = {}
            self.token_to_word = {}
            self.next_token = 0
            self.add_word('<unk>')

if __name__ == "__main__":
    # Demonstration usage
    print("Creating CustomTokenizer...")
    tokenizer = CustomTokenizer()
    
    # Tokenize a simple sentence
    print("\n1. Tokenizing first sentence...")
    sentence1 = "The cat sat on the mat"
    tokens1 = tokenizer.tokenize(sentence1)
    
    print(f"Input: '{sentence1}'")
    print(f"Tokens: {tokens1}")
    print(f"Vocabulary: {tokenizer.get_vocabulary()}")
    print(f"Vocabulary size: {tokenizer.get_vocabulary_size()}")
    
    # Tokenize a new sentence with a new word
    print("\n2. Tokenizing second sentence with new word...")
    sentence2 = "The dog ran"
    tokens2 = tokenizer.tokenize(sentence2)
    
    print(f"Input: '{sentence2}'")
    print(f"Tokens: {tokens2}")
    print(f"Updated vocabulary: {tokenizer.get_vocabulary()}")
    print(f"Updated vocabulary size: {tokenizer.get_vocabulary_size()}")
    
    # Detokenize to show reverse process
    print("\n3. Detokenizing tokens back to text...")
    reconstructed1 = tokenizer.detokenize(tokens1)
    reconstructed2 = tokenizer.detokenize(tokens2)
    
    print(f"Original: '{sentence1}'")
    print(f"Reconstructed: '{reconstructed1}'")
    print(f"Match: {sentence1 == reconstructed1}")
    
    print(f"Original: '{sentence2}'")
    print(f"Reconstructed: '{reconstructed2}'")
    print(f"Match: {sentence2 == reconstructed2}")
    
    # Test with unknown tokens
    print("\n4. Testing with unknown tokens...")
    unknown_tokens = [999, 1000]  # Tokens not in vocabulary
    unknown_text = tokenizer.detokenize(unknown_tokens)
    print(f"Unknown tokens: {unknown_tokens}")
    print(f"Detokenized: '{unknown_text}'")
    
    print("\nCustomTokenizer demonstration completed successfully!")
