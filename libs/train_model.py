import os

import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from libs.CityScapesDataset import CityscapesDataset

def train(model, epochs=30, lr=0.01, data_dir=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_save_path = os.path.join(os.getcwd(), 'results', 'models')
    os.makedirs(model_save_path, exist_ok=True)
    model_save_path = os.path.join(model_save_path, 'Semantic_Segmenter')

    train_data = CityscapesDataset(root_dir=data_dir, split="train")
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    for epoch in range(epochs):
        print(f'epoch:[{epoch+1}/{epochs}]')
        train_epoch(model, train_loader, criterion, optimizer, device)
        torch.save(model.state_dict(), model_save_path)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    t = tqdm(train_loader)
    for images, masks in t:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()
        t.set_description('(=> Training) Loss: {:.4f}'
                          .format(loss.item()))

