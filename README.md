AI-Generated Deepfake Detection
This project aims to detect deepfake facial content (images and video frames) using a fine-tuned ResNet18 deep learning model. It includes training scripts, dataset instructions, and a real-time web interface for evaluating image and video authenticity.

📌 Features
✅ Deepfake classification using a fine-tuned ResNet18 model

✅ Support for both image and video input

✅ Real-time inference via a Flask API and web interface

✅ ROC curve and performance metrics after training

✅ Clean and modular codebase with PyTorch and torchvision

📁 Dataset
This project uses two publicly available deepfake datasets from Kaggle.

🔗 Download Datasets
Deepfake with Cropped Faces from Video
👉 https://www.kaggle.com/datasets/pranabkc/deepfake-with-cropped-faces-from-video

Real and Fake Face Detection (CIPLab)
👉 https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

After downloading and extracting, arrange the data like this:

pgsql
Copy
Edit
AI-Generated-Deepfake-Detection/
└── dataset/
    ├── train/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
Update paths in train_video.py accordingly.

 Installation
Step 1: Clone the repository
bash
Copy
Edit
git clone https://github.com/Digeesha/AI-Generated-Deepfake-Detection.git
cd AI-Generated-Deepfake-Detection
Step 2: Create and activate a virtual environment
bash
Copy
Edit
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
Step 3: Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
🧠 Model Training
To train the model using video-based facial crops:

bash
Copy
Edit
cd model
python train_video.py
This script:

Extracts faces from videos

Trains a ResNet18 model

Saves the model (default: D:/deepfake-app/model/video_model.pth)

Outputs accuracy and plots an ROC curve

Ensure extract_faces in model/utils.py works correctly and paths are adjusted.

🌐 Run the Web Application
If provided, launch the web app using:

bash
Copy
Edit
cd app  # or the folder where app.py is located
python app.py
Open your browser at http://localhost:5000

Upload an image or video

Receive a classification: Real or Fake, along with a confidence score

📊 Example Performance (Test Set)
Model	Accuracy	Precision	Recall	F1-Score	AUC-ROC	Inference Time (ms)
ResNet18 (Proposed)	96.3%	95.8%	96.7%	96.2%	0.97	~40

📂 Folder Structure
bash
Copy
Edit
AI-Generated-Deepfake-Detection/
├── dataset/           # Downloaded data
├── model/             # Model scripts and training
│   └── train_video.py
├── app/               # Web app (Flask backend, optional frontend)
├── requirements.txt
└── README.md
📌 Future Work
Add SE blocks for attention-based feature refinement

Integrate LSTM or 3D CNNs for temporal modeling on videos

Improve adversarial robustness and model explainability

Extend to audio-visual deepfake detection

📧 Contact
For questions or feedback, reach out via GitHub Issues or contact the repository owner directly.

