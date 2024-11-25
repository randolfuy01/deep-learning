"""
Main script for running the training script for our data.
    - Defined dataloader
    - Defined model
    - Defined loss function
    - Defined optimer
    - Defined training loop
    - Defined validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
Predefined transformation for the data.
"""
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

"""
Define the dataset and the data loader.
"""
def get_dataloader(data_dir: str, batch_size: int, num_workers: int = 4):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return dataloader


class ConvNet(nn.Module):
    """
    ConvNet class defines the architecture of the Convolutional Neural Network
    for our model. We will be using 2 convolutional layers and 3 fully connected
    layers. The model will be trained on the dataset to classify the images into
    11 classes.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.dropout = nn.Dropout(0.5)

        # Placeholder for fc1 input size; will compute dynamically
        self._flattened_size = None

        # Fully connected layers
        self.fc1 = None
        self.fc2 = nn.Linear(90, 61)
        self.fc3 = nn.Linear(61, 11)

    def forward(self, x):
        """
        Forward pass of the network.
        Used to run the calculations necessary to classify the image.
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Dynamically calculate flattened size during first pass
        if self._flattened_size is None:
            self._flattened_size = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(self._flattened_size, 90)

        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

'''
Defined training loop for the model.
Components are interchangable based on our findings:
    - Loss function
    - Optimizer
    - Model
    - Device
'''
def training_loop(model, dataloader, num_epochs=20, learning_rate=1e-2, device="cpu"):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print(
                    f"Epoch {epoch + 1}, Iteration {i}, Loss: {running_loss / (i + 1):.4f}"
                )
        print(
            f"Epoch {epoch + 1} completed. Average Loss: {running_loss / len(dataloader):.4f}"
        )
    print("Finished Training!")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./dataset/Danger Of Extinction"
    batch_size = 32
    num_workers = 4
    num_epochs = 20
    train_loader = get_dataloader(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )
    for i, (inputs, labels) in enumerate(train_loader):
        print(f"Batch {i}: Inputs shape {inputs.shape}, Labels shape {labels.shape}")
        break
    model = ConvNet()
    training_loop(model, train_loader, num_epochs, 0.001, device)


if __name__ == "__main__":
    print("Begin script for convolutional model...")
    main()


