AI Plant Pathologist: Multi-Species Disease Detector
This project uses a Deep Learning model (ResNet18) to diagnose diseases in specific plant species. It features a graphical user interface (GUI) for image selection and a voice-enabled feedback system to announce results.

🎯 Supported Plants
The model is specifically trained to detect health issues in:

Strawberry

Tomato

Corn

Potato

✨ Key Features
AI Diagnosis: Uses a PyTorch-based ResNet18 architecture for high-accuracy leaf classification.

Voice Feedback: Integrated with pyttsx3 to speak out the diagnosis results.

Asynchronous Execution: Implements Multi-threading so the UI remains responsive while the voice engine is speaking.

Confidence Filtering: Only displays results if the model is more than 50% certain, ensuring reliable diagnoses.

📂 Project Structure
main.py: The primary Python script containing the Tkinter GUI and inference logic.

plant_disease_model.pth: The trained ResNet18 model weights.

class_names.json: Mapping of numeric model outputs to human-readable plant/disease labels.

🛠️ Setup & Installation
Requirements:

Run the Application:

🧠 How It Works
Input: You select a leaf image via a file dialog.

Processing: The image is resized to 224x224 and normalized using ImageNet standards.

Inference: The model runs on your CPU/GPU and calculates the probability of each disease class.

Output: A popup window shows the result, and a background thread speaks the diagnosis.