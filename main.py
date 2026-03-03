import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import pyttsx3      # The Voice Library
import threading    # The Multi-threading Library

# --- SETUP ---
MODEL_PATH = 'plant_disease_model.pth' # Ensure this file is in your folder
JSON_PATH = 'class_names.json'          # Ensure this file is in your folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Voice
engine = pyttsx3.init()

def speak_result(text):
    """Background thread function so the UI doesn't freeze"""
    engine.say(text)
    engine.runAndWait()

# --- REBUILD BRAIN ---
with open(JSON_PATH, 'r') as f:
    class_names = json.load(f)

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names)) 
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- INFERENCE PIPELINE ---
def upload_and_diagnose():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Plant Leaf Image")
    
    if not file_path: return

    img = Image.open(file_path).convert('RGB')
    img_t = data_transforms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        # Softmax converts raw math into probability %
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probs, 0)
    
    conf_percent = confidence.item() * 100
    label = class_names[predicted.item()].replace('___', ': ').replace('_', ' ')

    # --- CONFIDENCE THRESHOLDING ---
    if conf_percent >= 50.0: 
        result_text = f"Diagnosis: {label}. Confidence is {conf_percent:.1f} percent."
        print(result_text)
        
        # Start Voice Thread
        threading.Thread(target=speak_result, args=(result_text,)).start()
        messagebox.showinfo("AI Diagnosis", result_text)
    else:
        # Handling low-certainty detections
        error_msg = f"Uncertain. Confidence only {conf_percent:.1f}%. Please use a clearer photo."
        print(error_msg)
        threading.Thread(target=speak_result, args=("I am not sure. Please try again.",)).start()
        messagebox.showwarning("Low Confidence", error_msg)

if __name__ == "__main__":
    while True:
        upload_and_diagnose()
        if input("Scan again? (y/n): ").lower() != 'y': break