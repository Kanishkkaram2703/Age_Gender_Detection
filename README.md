# Age_Gender_Detection

An AI-powered web application that predicts age and gender from images using deep learning. Built with Streamlit, TensorFlow/Keras, and OpenCV, the app supports image uploads, snapshots from the camera, and real-time face detection via live webcam feed.

📌 Features
Upload one or more images and receive predictions

Take snapshots directly using your device’s webcam

Real-time face detection and prediction via live camera

Custom-trained CNN model for high accuracy

Smooth and responsive UI with custom styling

Error handling for file issues and camera access

Attribution footer: Powered by KANISHK KARAM 💡

🧠 How It Works
The model is a Convolutional Neural Network (CNN) trained to:

Predict Gender (binary classification: Male or Female)

Estimate Age (regression output as a number)

Preprocessing:

Convert image to grayscale

Resize to 128x128 pixels

Normalize pixel values (0 to 1)

Prediction Output:

Age is rounded to the nearest integer

Gender is classified based on a probability threshold (0.5)

Confidence score is shown for gender prediction

Live Camera Detection:

Uses Haar Cascade for face detection

Supports detection of multiple faces

Displays bounding boxes and labels in real time

📂 Project Structure
bash
Copy code
├── app.py                # Streamlit app script
├── best_model.h5         # Pre-trained Keras model
├── requirements.txt      # List of dependencies
└── README.md             # Project description
🚀 Getting Started
✅ Prerequisites
Python 3.7 or higher

pip for installing packages

🔧 Installation
bash
Copy code
pip install -r requirements.txt
▶️ Run the App
bash
Copy code
streamlit run app.py
🧪 Model Details
If you wish to train your own model:

Collect and label a dataset with face images and corresponding age/gender labels

Preprocess the dataset (resize, normalize, grayscale)

Use a CNN with:

A binary classification head for gender (sigmoid)

A regression head for age (linear)

Save the model as best_model.h5 using Keras

Author

Developed by KANISHK KARAM
Feel free to connect, contribute, or reach out!
