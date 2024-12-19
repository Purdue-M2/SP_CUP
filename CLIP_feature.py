import h5py 
import clip

from models.clip import clip
import torch
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import csv

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)
        return image

def extract(image_paths, batch_size=8, num_workers=8):
    device = 'cuda:3'
    model, preprocess = clip.load('ViT-L/14', device=device)
    model = model.to(device)

    dataset = ImageDataset(image_paths, preprocess)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    features = []

    with torch.no_grad():
        for batch_images in tqdm(data_loader, desc='extract'):
            batch_images_tensor = batch_images.to(device)
            batch_features = model.encode_image(batch_images_tensor)
            features.append(batch_features.cpu())

    features = torch.cat(features, dim=0)

    return features  

def get_names_from_csv(filename, column_name):
    image_paths = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)  # Use DictReader to handle the CSV as dictionaries
        for row in reader:
            image_paths.append(row[column_name])  # Access the column by name
    return image_paths 

if __name__ == '__main__':

    image_paths = get_names_from_csv('cup_val.csv','Image Path')


    features = extract(image_paths)
    print('features.shape =', features.shape)

    with h5py.File('cup_val.h5', 'w') as h5f:  # Open or create an HDF5 file
        # Create datasets within the HDF5 file
        h5f.create_dataset('test_features', data=features)
