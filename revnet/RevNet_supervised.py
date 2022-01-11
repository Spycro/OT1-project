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
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

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

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

writer = SummaryWriter()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # AlexNet
        # Image initiale de taille 96x96 avec 3 channels (r, g, b)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 2 * 2, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=4),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(model):
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(200):
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
supervised_model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=256 * 2 * 2, out_features=512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=512, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=10),
                nn.ReLU(),
            ).to(device)

print("Started training...")
train(supervised_model)
print("Finished training")
writer.close()