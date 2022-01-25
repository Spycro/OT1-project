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

# Test data (to report model performance)
test_set = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

BATCH_SIZE = 256

test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

writer = SummaryWriter()

model = torch.load("supervised_revnet.pth")

def test_model_accuracy(model):
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    correct = 0
    total = 0
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(test_dataloader):
            images = images.to(device)
            targets = targets.to(device)

            y = model(images)
            loss = criterion(y, targets)
            test_loss += loss.item()

            _, predicted = torch.max(y.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_loss = test_loss / len(test_dataloader)
    writer.add_scalar('Test/loss', test_loss)

    print(
        f"- Accuracy {correct / total * 100}%"
    )
    print(
        f"- Test Loss: {test_loss}"
    )

test_model_accuracy(model)
print("Finished")
writer.close()
