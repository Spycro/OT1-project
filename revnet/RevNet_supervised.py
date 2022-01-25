import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.manual_seed(42)

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

normalize = transforms.Normalize(mean.tolist(), std.tolist())
transform = transforms.Compose([transforms.ToTensor(), normalize])

# Train data
unlabeled_train_set = torchvision.datasets.STL10(root='./data', split='unlabeled', download=True, transform=transform)
labeled_train_set = torchvision.datasets.STL10(root='./data', split='train', folds=0, download=True, transform=transform)

# Test data (to report model performance)
test_set = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

# Split train set into a train and validation sets
labeled_train_size = int(0.80 * len(labeled_train_set))
valid_size = len(labeled_train_set) - labeled_train_size
train_set, valid_set = torch.utils.data.random_split(
    labeled_train_set, [labeled_train_size, valid_size]
)

BATCH_SIZE = 256

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

writer = SummaryWriter()

def train(model):
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for param in model.features.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True

    # Epoch 135: validation loss = 1.4138096570968628
    for epoch in range(135):
        print(
            f"Epoch {epoch+1} training"
        )
        train_loss = 0.0
        correct, total = 0, 0
        for images, targets in tqdm(train_dataloader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            y = model(images)
            loss = criterion(y, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(
            f"Epoch {epoch+1} validation"
        )
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, targets in tqdm(valid_dataloader):
                images = images.to(device)
                targets = targets.to(device)

                y = model(images)
                loss = criterion(y, targets)
                valid_loss += loss.item()

                _, predicted = torch.max(y.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        train_loss = train_loss / len(train_dataloader)
        valid_loss = valid_loss / len(valid_dataloader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Validation/train', valid_loss, epoch)

        print(
            f"- Accuracy {correct / total * 100}%"
        )
        print(
            f"- Training Loss: {train_loss}"
        )
        print(
            f"- Validation Loss: {valid_loss}"
        )


supervised_model = torch.load("pretrained_revnet.pth").to(device)
supervised_model.fc = nn.Linear(512, 10).to(device)

print("Started training...")
train(supervised_model)
torch.save(supervised_model, "supervised_revnet.pth")
print("Finished training")
writer.close()
