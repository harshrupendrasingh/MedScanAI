# 🧠 MedScanAI – Disease Detection from X-ray & CT Scans using CNN

**MedScanAI** is a deep learning-based diagnostic tool that uses **Convolutional Neural Networks (CNNs)** to detect diseases from medical images such as **X-rays** and **CT scans**. The project showcases how artificial intelligence can assist in early diagnosis and support medical professionals in clinical decision-making.

---

## 🚀 Features

- 🧠 CNN-based image classification model
- 🏥 Supports multiple imaging modalities (X-ray, CT)
- 📊 Performance metrics (accuracy, confusion matrix)
- 🔄 Easily extendable for other diseases or datasets
- 📁 Organized and modular codebase

---

## 📁 Project Structure

MedScanAI/
├── data/ # Dataset directory
├── models/ # Saved model weights
├── notebooks/ # Jupyter notebooks for experiments
├── src/ # Training, evaluation, model definition
├── utils/ # Helper functions (e.g., plots)
├── assets/ # Sample outputs, plots
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── main.py # Entry point


---

## 📦 Dataset

You can use any open-source medical imaging dataset, such as:

- [Chest X-ray (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Brain Tumor MRI](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

Place the dataset inside the `data/` directory.

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/MedScanAI.git
cd MedScanAI
pip install -r requirements.txt


Training
python main.py --mode train

Evaluation
bash
python main.py --mode eval

📈 Sample Results
Metric	Value
Accuracy	92.7%
F1 Score	92.3%

Results may vary depending on the dataset and configuration.

🧪 Sample Predictions
Image	Prediction
Pneumonia
Normal