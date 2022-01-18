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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAutocontrast(),
    transforms.RandomRotation(60),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop((96, 96))
])

# Train data
unlabeled_train_set = torchvision.datasets.STL10(root='./data', split='unlabeled', download=True, transform=transform)
labeled_train_set = torchvision.datasets.STL10(root='./data', split='train', folds=0, download=True, transform=transform)

# Test data (to report model performance)
test_set = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

# Split train set into a train and validation sets
unlabeled_train_size = int(0.80 * len(unlabeled_train_set))
valid_size = len(unlabeled_train_set) - unlabeled_train_size
train_set, valid_set = torch.utils.data.random_split(
    unlabeled_train_set, [unlabeled_train_size, valid_size]
)

BATCH_SIZE = 256

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

writer = SummaryWriter()

rotations_labels = [ 0, 90, 180, 270 ]

def generate_rotated_image_batch(batch_images):

    images = []
    targets = []
    
    for image in batch_images:
        images.append(image)
        images.append(transforms.functional.rotate(img=image, angle=90))
        images.append(transforms.functional.rotate(img=image, angle=180))
        images.append(transforms.functional.rotate(img=image, angle=270))
        targets.extend(range(4))

    images_tensor = torch.stack(images)
    targets_tensor = torch.tensor(targets)

    return images_tensor, targets_tensor

def train(model):
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    for epoch in range(15):
        print(
            f"Epoch {epoch+1} training"
        )
        train_loss = 0.0
        correct, total = 0, 0
        for batch_images, _ in tqdm(train_dataloader):
            [images, targets] = generate_rotated_image_batch(batch_images)

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
        for batch_images, _ in tqdm(valid_dataloader):
            [images, targets] = generate_rotated_image_batch(batch_images)

            images = images.to(device)
            targets = targets.to(device)

            y = model(images)
            loss = criterion(y, targets)
            valid_loss += loss.item()

            _, predicted = torch.max(y.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        scheduler.step()

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

print("Started training...")
pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False).to(device)
pretrained_model.fc = nn.Linear(512, 4).to(device)
train(pretrained_model)
torch.save(pretrained_model, "pretrained_revnet.pth")
print("Finished training")
writer.close()