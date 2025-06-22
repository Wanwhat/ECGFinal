import os

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from tqdm import tqdm

from dataset import ECGDataset
from seresnet import se_resnet34
from utils import set_seed, EarlyStopping, get_class_weights


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for data, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(data.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (outputs.argmax(dim=1) == labels.to(device)).sum().item() / labels.size(0)
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(val_loader):
            outputs = model(data.to(device))
            loss = criterion(outputs, labels.to(device))
            val_loss += loss.item()
            val_acc += (outputs.argmax(dim=1) == labels.to(device)).sum().item() / labels.size(0)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_path = './npy_files_2017/'
    train_dataset = ECGDataset(train_path)
    m = len(train_dataset)
    train_data, val_data = random_split(train_dataset, [m - int(0.2 * m), int(0.2 * m)],
                                        generator=torch.Generator().manual_seed(42))

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = se_resnet34(num_classes=2).to(device)

    class_weights = get_class_weights(train_data).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    check_points_path = './checkpoints/'
    if not os.path.exists(check_points_path):
        os.makedirs(check_points_path, exist_ok=True)
    early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.0001,
                                   path=os.path.join(check_points_path, 'best-model-2017.pt'))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 1000
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
