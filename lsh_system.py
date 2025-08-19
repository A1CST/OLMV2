import numpy as np
import hashlib
import pickle
import os
from collections import Counter

class LSHSystem:
    def __init__(self, latent_dim=128, num_hashes=64, checkpoint_path=None):
        """
        Initialize the LSH system with random hyperplanes
        
        Args:
            latent_dim (int): Dimension of the VAE's latent vector
            num_hashes (int): Number of hash functions (hyperplanes) to create
            checkpoint_path (str): Path to save/load the LSH state
        """
        self.latent_dim = latent_dim
        self.num_hashes = num_hashes
        self.checkpoint_path = checkpoint_path
        
        # Create random hyperplanes (the "sorter walls")
        # Shape: (latent_dim, num_hashes) - each column is a hyperplane
        self.hyperplanes = np.random.randn(latent_dim, num_hashes)
        
        # Normalize the hyperplanes for better LSH performance
        self.hyperplanes = self.hyperplanes / np.linalg.norm(self.hyperplanes, axis=0, keepdims=True)
        
        # Initialize state tracking attributes
        self.hash_counts = Counter()  # Track frequency of every hash
        self.all_hashes = set()       # Store every unique hash ever seen
        self.last_hash = None         # Track the most recently generated hash
        
        # Load previous state if checkpoint path is provided
        if self.checkpoint_path:
            self.load_state()
    
    def _get_vision_hash(self, latent_vector):
        """
        Generate LSH hash for vision data using random hyperplanes
        
        Args:
            latent_vector (numpy.ndarray): The latent vector from the VAE
            
        Returns:
            str: Binary string representing the LSH bucket ID
        """
        # Ensure latent_vector is a numpy array
        if not isinstance(latent_vector, np.ndarray):
            latent_vector = np.array(latent_vector)
        
        # Calculate dot product with all hyperplanes
        # Shape: (num_hashes,) - one value per hyperplane
        projections = np.dot(latent_vector, self.hyperplanes)
        
        # Determine sign of each projection
        # Positive or zero -> '1', negative -> '0'
        signs = (projections >= 0).astype(int)
        
        # Convert to binary string
        binary_string = ''.join(map(str, signs))
        
        return binary_string
    
    def generate_hash(self, sensory_packet, previous_tick_hash=None):
        """
        Generate a final hash combining all sensory data with LSH for vision
        
        Args:
            sensory_packet (dict): Dictionary containing sensory data
            previous_tick_hash (str or None): Hash from the previous tick
            
        Returns:
            str: Final SHA-256 hash as hexadecimal string
        """
        # Initialize data parts list
        data_parts = []
        
        # Add previous tick hash (or genesis string)
        if previous_tick_hash is None:
            data_parts.append("genesis")
        else:
            data_parts.append(previous_tick_hash)
        
        # Process Vision data with LSH
        if 'vision' in sensory_packet:
            try:
                latent_vector = sensory_packet['vision']['latent_vector']
                vision_hash = self._get_vision_hash(latent_vector)
                data_parts.append(f"vision:{vision_hash}")
            except (KeyError, TypeError) as e:
                # Fallback if latent_vector is not available
                data_parts.append(f"vision:error_{str(e)[:20]}")
        
        # Process Text data
        if 'text' in sensory_packet:
            text_data = sensory_packet['text']
            # Convert text data to string representation
            text_str = str(text_data)
            data_parts.append(f"text:{text_str}")
        
        # Process Keyboard data
        if 'keyboard' in sensory_packet:
            keyboard_data = sensory_packet['keyboard']
            # Convert keyboard data to string representation
            keyboard_str = str(keyboard_data)
            data_parts.append(f"keyboard:{keyboard_str}")
        
        # Combine all data parts with unique delimiter
        combined_string = "|".join(data_parts)
        
        # Encode to bytes and compute SHA-256 hash
        combined_bytes = combined_string.encode('utf-8')
        final_hash = hashlib.sha256(combined_bytes).hexdigest()
        
        # Update state tracking attributes
        self.all_hashes.add(final_hash)
        self.hash_counts[final_hash] += 1
        self.last_hash = final_hash
        
        return final_hash
    
    def get_vision_similarity(self, hash1, hash2):
        """
        Calculate similarity between two vision hashes (Hamming distance)
        
        Args:
            hash1 (str): First binary hash string
            hash2 (str): Second binary hash string
            
        Returns:
            float: Similarity score (0.0 = identical, 1.0 = completely different)
        """
        if len(hash1) != len(hash2):
            raise ValueError("Hash strings must have the same length")
        
        # Calculate Hamming distance
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        
        # Convert to similarity score (0.0 = identical, 1.0 = completely different)
        similarity = hamming_distance / len(hash1)
        
        return similarity
    
    def get_novel_hashes(self, new_hashes):
        """
        Get hashes that are not already in the system's history
        
        Args:
            new_hashes (list): List of hash strings to check
            
        Returns:
            list: Subset of new_hashes containing only novel hashes
        """
        return [hash_val for hash_val in new_hashes if hash_val not in self.all_hashes]
    
    def get_general_hash(self):
        """
        Get the most common hash from the system's history
        
        Returns:
            str or None: The most frequent hash, or None if no hashes exist
        """
        if not self.hash_counts:
            return None
        
        return self.hash_counts.most_common(1)[0][0]
    
    def get_master_hash(self):
        """
        Get the master hash - the most common hash that is also the last hash
        
        Returns:
            str or None: The master hash if conditions are met, otherwise None
        """
        general_hash = self.get_general_hash()
        
        if general_hash is None or self.last_hash is None:
            return None
        
        # Check if the most common hash is also the last hash
        if general_hash == self.last_hash:
            return general_hash
        
        return None
    
    def load_state(self):
        """
        Load the LSH system's state from the checkpoint file
        """
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return
        
        try:
            with open(self.checkpoint_path, 'rb') as f:
                state_data = pickle.load(f)
                
            # Update instance attributes with loaded data
            self.hash_counts = Counter(state_data.get('hash_counts', {}))
            self.all_hashes = set(state_data.get('all_hashes', set()))
            self.last_hash = state_data.get('last_hash', None)
            
        except (pickle.PickleError, IOError, KeyError) as e:
            # Handle corrupted or invalid checkpoint files gracefully
            print(f"Warning: Could not load LSH state from {self.checkpoint_path}: {e}")
            # Reset to default state
            self.hash_counts = Counter()
            self.all_hashes = set()
            self.last_hash = None
    
    def save_state(self):
        """
        Save the LSH system's state to the checkpoint file
        """
        if not self.checkpoint_path:
            return
        
        try:
            # Package current state into a dictionary
            state_data = {
                'hash_counts': dict(self.hash_counts),
                'all_hashes': self.all_hashes,
                'last_hash': self.last_hash
            }
            
            # Ensure the directory exists (only if there's a directory component)
            directory = os.path.dirname(self.checkpoint_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Save to file
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(state_data, f)
                
        except (pickle.PickleError, IOError) as e:
            print(f"Error: Could not save LSH state to {self.checkpoint_path}: {e}")

if __name__ == "__main__":
    # Demonstration usage
    print("Creating LSHSystem...")
    lsh = LSHSystem(latent_dim=128, num_hashes=64, checkpoint_path="test_lsh_memory.pkl")
    
    # Create dummy sensory packet
    print("\n1. Testing with dummy sensory packet...")
    dummy_latent = np.random.randn(128)
    dummy_packet = {
        'vision': {
            'latent_vector': dummy_latent,
            'image_size': (1920, 1080)
        },
        'text': {
            'source': 'user',
            'text': 'Hello world'
        },
        'keyboard': {
            'type': 'keyboard_utterance',
            'events': [{'type': 'press', 'key': 'h'}]
        }
    }
    
    # Generate hash
    hash1 = lsh.generate_hash(dummy_packet, None)
    print(f"Generated hash: {hash1[:32]}...")
    
    # Test with similar vision data
    print("\n2. Testing with similar vision data...")
    similar_latent = dummy_latent + np.random.normal(0, 0.1, 128)  # Add small noise
    similar_packet = {
        'vision': {
            'latent_vector': similar_latent,
            'image_size': (1920, 1080)
        }
    }
    
    hash2 = lsh.generate_hash(similar_packet, hash1)
    print(f"Similar hash: {hash2[:32]}...")
    
    # Test vision similarity
    vision_hash1 = lsh._get_vision_hash(dummy_latent)
    vision_hash2 = lsh._get_vision_hash(similar_latent)
    similarity = lsh.get_vision_similarity(vision_hash1, vision_hash2)
    print(f"Vision similarity: {similarity:.3f} (0.0 = identical, 1.0 = completely different)")
    
    # Test with very different vision data
    print("\n3. Testing with different vision data...")
    different_latent = np.random.randn(128)
    different_packet = {
        'vision': {
            'latent_vector': different_latent,
            'image_size': (1920, 1080)
        }
    }
    
    hash3 = lsh.generate_hash(different_packet, hash2)
    print(f"Different hash: {hash3[:32]}...")
    
    # Test vision similarity with different data
    vision_hash3 = lsh._get_vision_hash(different_latent)
    similarity_diff = lsh.get_vision_similarity(vision_hash1, vision_hash3)
    print(f"Different vision similarity: {similarity_diff:.3f}")
    
    # Test state management functionality
    print("\n4. Testing state management...")
    print(f"Total unique hashes: {len(lsh.all_hashes)}")
    print(f"Hash counts: {dict(lsh.hash_counts)}")
    print(f"Last hash: {lsh.last_hash[:16]}...")
    
    # Test getter methods
    general_hash = lsh.get_general_hash()
    print(f"General hash: {general_hash[:16] if general_hash else 'None'}...")
    
    master_hash = lsh.get_master_hash()
    print(f"Master hash: {master_hash[:16] if master_hash else 'None'}...")
    
    # Test novel hash detection
    test_hashes = [hash1, "new_hash_123", hash2]
    novel_hashes = lsh.get_novel_hashes(test_hashes)
    print(f"Novel hashes from test set: {len(novel_hashes)}")
    
    # Save state
    print("\n5. Saving state...")
    lsh.save_state()
    print("State saved successfully!")
    
    # Test loading state
    print("\n6. Testing state loading...")
    lsh2 = LSHSystem(latent_dim=128, num_hashes=64, checkpoint_path="test_lsh_memory.pkl")
    print(f"Loaded state - Total unique hashes: {len(lsh2.all_hashes)}")
    print(f"Loaded state - Hash counts: {dict(lsh2.hash_counts)}")
    
    # Clean up test file
    if os.path.exists("test_lsh_memory.pkl"):
        os.remove("test_lsh_memory.pkl")
        print("Test file cleaned up.")
    
    print("\nLSHSystem demonstration completed successfully!")
