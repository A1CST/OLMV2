import os
import pickle

class HashInfoDB:
    def __init__(self, checkpoint_path=None):
        """
        Initialize the Hash Information Database.

        Args:
            checkpoint_path (str): Path to save/load the database state.
        """
        self.checkpoint_path = checkpoint_path
        
        # Main database mapping hash_id to its metadata dictionary
        self.db = {}
        
        # Mappings for quick lookups
        self.hash_to_id = {}
        self.next_id = 0
        
        if self.checkpoint_path:
            self.load_state()

    def get_or_create_hash_info(self, hash_string):
        """
        Retrieve info for a hash, creating a new entry if it's the first time.

        Args:
            hash_string (str): The full SHA-256 hash.

        Returns:
            tuple: A tuple containing (hash_id, hash_info, is_new)
                   - hash_id (int): The short ID for the hash.
                   - hash_info (dict): The metadata for the hash.
                   - is_new (bool): True if the hash was just added, False otherwise.
        """
        if hash_string in self.hash_to_id:
            hash_id = self.hash_to_id[hash_string]
            return hash_id, self.db[hash_id], False
        else:
            # This is a new, never-before-seen hash
            hash_id = self.next_id
            self.hash_to_id[hash_string] = hash_id
            
            # Create the initial metadata structure
            self.db[hash_id] = {
                'hash_string': hash_string,
                'state_impact': {
                    # Lists to store observed changes, e.g., [-2, -3, -2.5]
                    'Novelty': [],
                    'Energy': [],
                    'Boredom': [],
                    'Comfort': [],
                },
                'optimal_depth': None,      # To be determined by calibration
                'streamlined_depth': None,  # To be determined by calibration
                'deep_depth': None,         # To be determined by calibration
            }
            
            self.next_id += 1
            return hash_id, self.db[hash_id], True

    def update_state_impact(self, hash_id, state_changes):
        """
        Update the state_impact record for a given hash_id.

        Args:
            hash_id (int): The ID of the hash to update.
            state_changes (dict): A dict of changes, e.g., {'Comfort': -2, 'Novelty': 3}.
        """
        if hash_id not in self.db:
            return
            
        for key, value in state_changes.items():
            if key in self.db[hash_id]['state_impact']:
                self.db[hash_id]['state_impact'][key].append(value)

    def update_depths(self, hash_id, depths):
        """
        Update the calibrated depths for a given hash_id.

        Args:
            hash_id (int): The ID of the hash to update.
            depths (dict): A dict with keys 'optimal', 'streamlined', 'deep'.
        """
        if hash_id not in self.db:
            return
            
        self.db[hash_id]['optimal_depth'] = depths.get('optimal')
        self.db[hash_id]['streamlined_depth'] = depths.get('streamlined')
        self.db[hash_id]['deep_depth'] = depths.get('deep')

    def save_state(self):
        """Persist the database to disk."""
        if not self.checkpoint_path:
            return
        
        state = {
            'db': self.db,
            'hash_to_id': self.hash_to_id,
            'next_id': self.next_id,
        }
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Error saving HashInfoDB state: {e}")

    def load_state(self):
        """Load the database from disk."""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return
        try:
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            self.db = state.get('db', {})
            self.hash_to_id = state.get('hash_to_id', {})
            self.next_id = state.get('next_id', 0)
        except Exception as e:
            print(f"Error loading HashInfoDB state: {e}")
