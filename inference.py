"""
    DEMO FOR CNN INFERENCE PIPELINE
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_architecture import ConvNet
from collections import Counter
import cv2

animals = {
    0: "elephant",
    1: "amur leopard",
    2: "arctic fox",
    3: "chimpanzee",
    4: "jaguar",
    5: "lion",
    6: "orangutan",
    7: "panda",
    8: "panther",
    9: "rhino",
    10: "cheetah",
}

class InferencePipeline:
    def __init__(self, model_path):
        """
        Initialize the inference pipeline with the pre-trained model and transformations.
        """
        self.model = self.load_model(model_path)
        self.image_transform = self.define_image_transform()
        self.frame_transform = self.define_frame_transform()

    @staticmethod
    def load_model(model_path):
        """
        Load the pre-trained CNN model for inference.
        """
        model = ConvNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  
        return model

    @staticmethod
    def define_image_transform():
        """
        Define the transformation pipeline for images.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def define_frame_transform():
        """
        Define the transformation pipeline for video frames.
        """
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224)), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image_path):
        """
        Preprocess an image for input into the CNN model.
        """
        try:
            image = Image.open(image_path).convert("RGB")  
            return self.image_transform(image).unsqueeze(0) 
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")

    def preprocess_frame(self, frame):
        """
        Preprocess a single video frame for input into the CNN model.
        """
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frame_tensor = self.frame_transform(frame)  
            return frame_tensor.unsqueeze(0) 
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None

    def infer(self, tensor):
        """
        Perform inference on a preprocessed tensor (frame or image).
        """
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

    def infer_image(self, image_path):
        """
        Perform inference on a single image.
        """
        image_tensor = self.preprocess_image(image_path)
        return self.infer(image_tensor)

    def process_video(self, video_path, frame_rate=1):
        """
        Process a video, classify each frame, and vote on the final label.
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps // frame_rate

        predictions = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_tensor = self.preprocess_frame(frame)
                if frame_tensor is not None:
                    prediction = self.infer(frame_tensor)
                    predictions.append(prediction)

            frame_count += 1

        cap.release()

        if not predictions:
            raise ValueError("No frames were processed. Please check the video or preprocessing pipeline.")

        return Counter(predictions).most_common(1)[0][0]


def main():
    model_path = './cnn_model_test_4.pth'
    pipeline = InferencePipeline(model_path)

    # Inference on images
    # img_paths = ["./imgs/cheetah.png", "./imgs/panda.png", "./imgs/panther.png"]
    # for img_path in img_paths:
    #     prediction = pipeline.infer_image(img_path)
    #     print(f"Final Predicted Class for {img_path}: {animals[prediction]}")
        
    # Inference on videos
    video_paths = ["./vids/elephant.mp4", "./vids/lion.mp4", "./vids/artic_fox.mp4", "./vids/rhino.mp4"]
    for video_path in video_paths:
        prediction = pipeline.process_video(video_path)
        print(f"Final Predicted Class for {video_path}: {animals[prediction]}")


if __name__ == "__main__":
    main()
