"""
The script uses a Convolutional Neural Network model for image classification on endangered species.
Images on the dataset are classified into 11 different categories based on the species. There is no specific
image sizing for the dataset  so the images are resized to 224x224 pixels. The model uses 4 convolutional layers
and 3 fully connected layers. The model is initially trained using the Adam optimizer and the CrossEntropyLoss function.
The training loop runs for 50 epochs with a learning rate of 0.001. 

Availablility on Devices:
    Can be run on different devices (cpu, cuda, mps) based on availability.
    For fastest/best results, use a GPU.
    
Note: This script is a template and can be modified based on the desired results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


"""
Predefined transformation for the data.
"""
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_dataloader(transform, root, batch_size, split_ratio=0.8, num_workers=2):
    """
    Creates training and testing data loaders by splitting the dataset.

    Args:
        transform (transforms.Compose): A composition of transformations to apply to the images.
        root (str): The root directory containing the dataset.
        batch_size (int): The number of samples per batch.
        split_ratio (float, optional): The proportion of the dataset to use for training. Defaults to 0.8.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 2.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader and the testing DataLoader.
    """
    
    dataset = datasets.ImageFolder(root=root, transform=transform)
    train_size = int(len(dataset) * split_ratio)
    valid_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - valid_size
    
    trainset, validset, testset = random_split(dataset, [train_size, valid_size, test_size])
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, validloader, testloader


class ConvNet(nn.Module):
    """
    Convolutional Neural Network model for image classification on endangered species.
    4 convolutional layers and 3 fully connected layers.
        - More layers to better caputure 

    Args:
        nn (_type_): _description_
    """    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 11)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def training_loop(model, trainloader, validloader, num_epochs=20, learning_rate=0.001, device="cpu"):
    """
    Training loop for training the neural network.
    Changes can be made to the loss function, optimizer, model, and device.

    Args:
        model (_type_): The type of model to train
        dataloader (_type_): The type of dataloader to use
        num_epochs (int, optional): The number of training cyles to do. Defaults to 20.
        learning_rate (_type_, optional):The rate at which we can perform gradient descent. Defaults to 1e-2.
        device (str, optional): Device to use for training (preferably gpu if possible). Defaults to "cpu".
    """    
    
    '''
    Change the loss function and optimizer based on our findings.
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(trainloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.3f}')
        

        validate(model, validloader, criterion, device)

def validate(model, dataloader, criterion, device):
    """
    Validation function to evaluate model performance after each epoch

    Args:
        model (_type_): model we want to validate
        dataloader (_type_): validation data we are using
        criterion (_type_): loss function we are using to evaluate the model
        device (_type_): device we are using for training (mps, cuda, cpu)
    """    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
    model.train()

def main():
    
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else 
        torch.device("cuda") if torch.cuda.is_available() else 
        torch.device("cpu")
    )
    print("Device: ", device)
    
    """
    Changable parameters for the model.
    Change based on desired results.
    """
    root = "./dataset/Danger Of Extinction"
    batch_size = 32
    num_workers = 4
    num_epochs = 50
    split_ratio = 0.8
    learning_rate = 0.001
    
    print("Getting dataloader...")
    train_loader, valid_loader, test_loader = get_dataloader(
        transform=transform, root=root, batch_size=batch_size, split_ratio=split_ratio, num_workers=num_workers
    )
    
    model = ConvNet().to(device)
    
    print("Starting training loop...")
    training_loop(
        model=model, 
        trainloader=train_loader, 
        validloader=valid_loader,
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        device=device
    )
    
    # Testing the model on the test dataset
    print("Testing the model on the test dataset...")
    validate(model, test_loader, nn.CrossEntropyLoss(), device)



if __name__ == "__main__":
    print("Begin script for convolutional model...")
    
    main()