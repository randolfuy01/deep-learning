import torch
from PIL import Image
from torchvision import transforms, datasets
from main import ConvNet
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
import matplotlib.image as mpimg

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def predict_image(model, image_path, device="cpu"):
    """
    Make a prediction on a single image.

    Args:
        model: Trained model to use for prediction.
        image_path (str): Path to the image file.
        device (str): The device to run the model on. Defaults to "cpu".

    Returns:
        predicted_class: The predicted class of the image.
    """

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  
    image = image.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1) 

    return predicted.item()


if __name__ == "__main__":
    model = ConvNet() 
    model.load_state_dict(torch.load("./model/cnn_model_test_3.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for i in range(1,23):
            image_path = f"./test/test{i}.jpg"
            img = mpimg.imread(image_path)

            dataset = datasets.ImageFolder("./dataset/Danger Of Extinction", transform=transform)
            predicted = predict_image(model, image_path, device)
            print(f"test{i}.jpg")
            
            class_name = dataset.classes[predicted]
            print(f"Predicted to be: {class_name}")

            plt.imshow(img)
            plt.title(f"Predicted to be: {class_name}")
            plt.show()