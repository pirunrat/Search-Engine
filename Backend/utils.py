import pickle
import torch
from skipgram_model import Skipgram

class Utils:
    def __init__(self, path) -> None:
        self.path = path

    def load(self):
        with open(self.path, 'rb') as f:
            loaded_variable = pickle.load(f)
        return loaded_variable
    
    def load_pytorch_model(self):
        try:
            voc_size = 2411
            embedding_size = 2
                # Create a new instance of the Skipgram model
            loaded_model = Skipgram(voc_size, embedding_size)

            # Load the saved model parameters into the new instance
            loaded_model.load_state_dict(torch.load(self.path))

             # Set the model to evaluation mode
            loaded_model.eval()
            print("Model has been loaded successfully")
            return loaded_model
        except Exception as e:
            print(f"Error loading PyTorch model from {self.path}: {e}")
            return None