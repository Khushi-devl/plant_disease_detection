import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

# --- 1. SETTING PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model.pth')
JSON_PATH = os.path.join(BASE_DIR, 'class_indices.json')

# Voice Setup
try:
    import pyttsx3
    engine = pyttsx3.init()
except:
    engine = None

def speak_result(text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except: pass

# --- 2. LOAD DATA & MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(JSON_PATH):
    print(f"❌ Error: {JSON_PATH} nahi mili!")
    exit()

with open(JSON_PATH, 'r') as f:
    class_indices = json.load(f)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names)) 

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ System Ready!")
except Exception as e:
    print(f"❌ Model load error: {e}")
    exit()

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. DIAGNOSIS FUNCTION (Isme badlao kiya hai) ---
def upload_and_diagnose():
    file_path = filedialog.askopenfilename(title="Select Plant Leaf Image")
    if not file_path: return

    try:
        img = Image.open(file_path).convert('RGB')
        img_t = data_transforms(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_t)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted = torch.max(probs, 0)
        
        accuracy = confidence.item() * 100
        
        # --- NAYA LOGIC START ---
        THRESHOLD = 80.0  # Agar 80% se kam sure hai toh mana kar do
        
        if accuracy < THRESHOLD:
            result_text = "❌ Not Able to Recognize\n(Plant not in dataset or unclear photo)"
            voice_text = "I am not able to recognize this. Please try a clearer photo or a different plant."
            print(f"\n⚠️ Low Confidence: {accuracy:.2f}% (Rejected)")
        else:
            raw_label = class_names[predicted.item()]
            if "___" in raw_label:
                plant, disease = raw_label.split("___")
            else:
                plant, disease = "Unknown", raw_label

            plant = plant.replace('_', ' ')
            disease = disease.replace('_', ' ')
            result_text = f"Plant: {plant}\nDisease: {disease}\nAccuracy: {accuracy:.2f}%"
            voice_text = f"The plant is {plant}. The disease is {disease}."
            print(f"\n✅ Recognized: {plant} ({accuracy:.2f}%)")
        # --- NAYA LOGIC END ---

        print("-" * 30 + "\n" + result_text + "\n" + "-" * 30)
        threading.Thread(target=speak_result, args=(voice_text,)).start()
        messagebox.showinfo("AI Result", result_text)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    while True:
        upload_and_diagnose()
        if input("\nScan another photo? (y/n): ").lower() != 'y':
            break
    root.destroy()