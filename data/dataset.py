from torch.utils.data import Dataset
from .preprocess import Preprocessor

#Dataset class
class SER_Dataset(Dataset):
    def __init__(self, preprocessor:Preprocessor = None):
        self.preprocessor = preprocessor #preprocessor function to preprocess audios
    
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass