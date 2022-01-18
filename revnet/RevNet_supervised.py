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
labeled_train_size = int(0.48 * len(labeled_train_set))
labeled_valid_size = int(0.60 * len(labeled_train_set)) - labeled_train_size
remaining_size = len(labeled_train_set) - (labeled_train_size + labeled_valid_size)
labeled_finetune_train_size = int(0.32 * remaining_size)
labeled_finetune_valid_size = remaining_size - labeled_finetune_train_size
train_set, valid_set, finetune_train_set, finetune_valid_set = torch.utils.data.random_split(
    labeled_train_set, [labeled_train_size, labeled_valid_size, labeled_finetune_train_size, labeled_finetune_valid_size]
)

BATCH_SIZE = 64

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
finetune_train_dataloader = DataLoader(finetune_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
finetune_valid_dataloader = DataLoader(finetune_valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

writer = SummaryWriter()

def train(model, train_dataloader, valid_dataloader, writer_prefix):
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
        writer.add_scalar(writer_prefix + '/Loss/train', train_loss, epoch)
        writer.add_scalar(writer_prefix + '/Validation/train', valid_loss, epoch)

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

# Freeze all layers except classifier
for name, param in supervised_model.named_parameters():
    param.requires_grad = False

# Last layer size of ResNet should be 512
supervised_model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=10),
                nn.ReLU(),
            ).to(device)


print("Started training in freezed model.")
train(supervised_model, train_dataloader, valid_dataloader, 'Freezed')
print("Started fine-tuning on model.")

# Unfreeze all layers
for name, param in supervised_model.named_parameters():
    param.requires_grad = True

train(supervised_model, finetune_train_dataloader, finetune_valid_dataloader, 'Finetune')
print("Finished training")
writer.close()