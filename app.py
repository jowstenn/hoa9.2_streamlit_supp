import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# load the class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# load the model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("alien_predator_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# transform data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("Alien vs Predator Classifier")

file = st.file_uploader("Upload an image", type=["jpg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    st.success(f"Prediction: {class_names[pred.item()]}")
