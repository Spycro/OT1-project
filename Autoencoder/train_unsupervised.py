import torchvision
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms
from autoencoderResnet import AutoEncoder
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Hyperparameters
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "gpu"
BATCH_SIZE = 512
EPOCHS = 500
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
PLOT_POINTS = False

LOAD_MODEL_FILE = ""
LOG_DIR = "logs/Autoencoder/archi1/v1"


def train_autoencoder(train_loader, model, optimizer, loss_fn, epoch, writer):
    """Method to train the model

    Args:
        train_loader (DataLoader): dataloader with the pictures
        model (Module): net architecture
        optimizer (Optimizer): optimizer used on the model
        loss_fn (Module): loss function
        epoch (int): epoch number 
        writer (SummaryWriter): Tensorboard's writer to plot metrics
        scheduler (Scheduler): Scheduler used to reduce lr
    """
    loop = tqdm(train_loader, leave=True)
    train_loss = []
    model.train()

    # plot lr on tensorboard
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
    countTotal = len(train_loader.dataset)/BATCH_SIZE

    for batch_idx, (data, _) in enumerate(loop):
        data = data.to(DEVICE)
        output = model(data)
        loss = loss_fn(output, data)
        train_loss.append(loss.item()/countTotal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update Progress bar
        loop.set_postfix(loss=loss.item()/countTotal, epoch=batch_idx)

    # Plot training loss on tensorboard
    print(f"Mean Train loss : {sum(train_loss)/len(train_loss)}")

    writer.add_scalar("train loss", sum(train_loss)/len(train_loss), epoch)
    return sum(train_loss)/len(train_loss)


def val_autoencoder(val_loader, model, loss_fn, epoch, writer):
    """Method to train the model

    Args:
        val_loader (DataLoader): dataloader with the pictures
        model (Module): net architecture
        loss_fn (Module): loss function
        epoch (int): epoch number 
        writer (SummaryWriter): Tensorboard's writer to plot metrics
    """
    loop = tqdm(val_loader, leave=True)
    val_loss = []
    model.eval()
    countTotal = len(val_loader.dataset)/BATCH_SIZE

    for batch_idx, (data, _) in enumerate(val_loader):
        data = data.to(DEVICE)
        output = model(data)
        data = data.data.cpu().detach()
        output = output.data.cpu().detach()

        if(PLOT_POINTS == True):
            for i in range(output.shape[0]):
                f, axarr = plt.subplots(2, 2)

                img = data[i].permute(1, 2, 0)
                out = output[i].permute(1, 2, 0)

                axarr[0, 0].imshow(out)
                axarr[0, 0].set_title("output")

                axarr[1, 0].imshow(img)
                axarr[1, 0].set_title("target")
                plt.show()

        loss = loss_fn(output, data)
        val_loss.append(loss.item()/countTotal)
        # Update Progress bar
        loop.set_postfix(loss=loss.item()/countTotal, epoch=batch_idx)

    # Plot training loss on tensorboard
    print(f"Mean Val loss : {sum(val_loss)/len(val_loss)}")

    writer.add_scalar("val loss", sum(val_loss)/len(val_loss), epoch)
    return sum(val_loss)/len(val_loss)


def train_val_test(dataset, ratio_val, ratio_test):
    """Split dataset into train,validation and test dataset

    Args:
        dataset (Dataset): Dataset to split
        ratio_val (float): Ratio for the validation
        ratio_test (float): Ratio for the test

    Returns:
        Map: Map with the 3 datasets 
    """
    training_size = int(len(dataset)*(1-ratio_val-ratio_test))
    val_size = int(len(dataset)*(ratio_val))
    train_idx, val_idx, test_idx = random_split(dataset, [training_size, val_size, len(
        dataset)-val_size-training_size], generator=torch.Generator().manual_seed(42))
    datasets = {}
    datasets['train'] = train_idx
    datasets['val'] = val_idx
    datasets['test'] = test_idx
    return datasets


def main():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    unlabeled_train_set = torchvision.datasets.STL10(
        root='./data', split='unlabeled', download=False, transform=transform)
    labeled_train_set = torchvision.datasets.STL10(
        root='./data', split='train', folds=0, download=False, transform=transform)

    # Test data (to report model performance)
    test_set = torchvision.datasets.STL10(
        root='./data', split='test', download=False)

    datasets = train_val_test(unlabeled_train_set, 0.2, 0.0)

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE)
    val_loader = DataLoader(datasets["val"], batch_size=BATCH_SIZE)

    loss_ae = torch.nn.MSELoss()

    # Initialize the tensorboard's writer
    writer = SummaryWriter(
        log_dir=LOG_DIR)

    # Defines or load the model
    if LOAD_MODEL:
        model = torch.load(LOAD_MODEL_FILE).to(DEVICE)
    else:
        model = AutoEncoder(512).to(DEVICE)
        # Defines the optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    best_loss = val_autoencoder(val_loader, model, loss_ae, 0, writer)

    for epochs in range(EPOCHS):
        train_loss = train_autoencoder(
            train_loader, model, optimizer, loss_ae, epochs, writer)

        val_loss = val_autoencoder(val_loader, model, loss_ae, epochs, writer)

        if(val_loss < best_loss):
            best_loss = val_loss
            torch.save(model, "ae_"+str(epochs))


if __name__ == "__main__":
    main()
