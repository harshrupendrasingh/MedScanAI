
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from src.model import SimpleCNN
from src.utils import get_device

device = get_device()
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

model = SimpleCNN(num_classes=len(class_names))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict(image):
    image = Image.fromarray(image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    
    top_class = class_names[np.argmax(probs)]
    conf = np.max(probs) * 100
    return {cls: float(f"{p*100:.2f}") for cls, p in zip(class_names, probs)}, f"Predicted: {top_class} ({conf:.2f}%)"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray"),
    outputs=[
        gr.Label(num_top_classes=4, label="Prediction Probabilities"),
        gr.Textbox(label="Result")
    ],
    title="MedScanAI – Chest X-ray Disease Classifier",
    description="Upload a chest X-ray to detect COVID, Viral Pneumonia, Lung Opacity, or Normal cases."
)

interface.launch()

