import torch
from torch import nn
from torch.utils.data import random_split
from tqdm import tqdm

from dataset import ECGDataset
from seresnet import se_resnet34
from utils import set_seed, EarlyStopping, get_class_weights


def load_pretrained_model(model_path, num_classes, device):
    model = se_resnet34(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if list(checkpoint.keys())[0].startswith('module.'):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=False)

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True

    if hasattr(model, 'adaptiveavgpool') and hasattr(model, 'adaptivemaxpool'):
        for param in model.adaptiveavgpool.parameters():
            param.requires_grad = True
        for param in model.adaptivemaxpool.parameters():
            param.requires_grad = True

    for name, module in model.named_modules():
        if 'se' in name.lower():
            for param in module.parameters():
                param.requires_grad = True

    return model.to(device)


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

    pretrained_model_path = './checkpoints/best-model-2017.pt'
    finetuned_model_path = './checkpoints/best-model.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_pretrained_model(pretrained_model_path, num_classes=2, device=device)

    train_path = './data_labeled/'
    train_dataset = ECGDataset(train_path)
    m = len(train_dataset)
    train_data, val_data = random_split(train_dataset, [m - int(0.2 * m), int(0.2 * m)],
                                        generator=torch.Generator().manual_seed(42))

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    class_weights = get_class_weights(train_data).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.0001,
                                   path=finetuned_model_path)

    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    for epoch in range(1000):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1} -- '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
