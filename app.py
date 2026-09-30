import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================
# MODEL PATH
# =========================
MODEL_PATH = "C:/Users/SWARA/Downloads/AIDSML SEM 4 Project/best_model.pth"

# =========================
# CLASS LABELS (43 classes)
# =========================
classes = {
    0: "Speed limit (20 km/h)",
    1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)",
    5: "Speed limit (80 km/h)",
    6: "End speed limit (80 km/h)",
    7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)",
    9: "No passing",
    10: "No passing (>3.5 tons)",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles >3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Ice/snow warning",
    31: "Wild animals crossing",
    32: "End restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Straight or right",
    37: "Straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End no passing",
    42: "End no passing (>3.5 tons)"
}

# =========================
# MODEL ARCHITECTURE
# =========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Linear(256, 43)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# =========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    model = CNN()
    state_dict = torch.load("best_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
model = load_model()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # MUST match training
    transforms.ToTensor()
])

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Road Sign Intelligence System")

st.title("Road Sign Intelligence System")
st.write("Upload a road sign image and get prediction (43 classes).")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    if st.button("Predict"):

        with st.spinner("Analyzing road sign..."):
            with torch.no_grad():
                output = model(img)
                probs = torch.softmax(output, dim=1)

                top_prob, top_class = torch.topk(probs, 3)

                pred = top_class[0][0].item()
                conf = top_prob[0][0].item()

        st.success(f"Prediction: {classes[pred]}")
        st.info(f"Confidence: {conf * 100:.2f}%")

        st.markdown("### 🔍 Top 3 Predictions")

        for i in range(3):
            idx = top_class[0][i].item()
            prob = top_prob[0][i].item()
            st.write(f"{classes[idx]} → {prob * 100:.2f}%")