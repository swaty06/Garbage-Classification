# 🗑️ Garbage Classification with ResNet

A deep learning-based image classification project that uses **ResNet** to classify different types of garbage with **94% accuracy**. This project aims to support better waste sorting by automatically recognizing and classifying garbage into 10 distinct categories.

## 🚀 Overview

This model helps in automating the classification of waste materials into the following classes:

- **Battery**
- **Biological**
- **Cardboard**
- **Clothes**
- **Glass**
- **Metal**
- **Paper**
- **Plastic**
- **Shoes**
- **Trash**

Such a model can be helpful for waste management systems, recycling centers, or educational tools for sustainability.

## 🧠 Model Architecture

The classifier is built using a **ResNet** architecture (Residual Neural Network), which is known for its performance in image recognition tasks. The model was trained on a labeled dataset of garbage images and achieved **94% accuracy** on the test set.

## 🛠️ Technologies Used

- Python
- PyTorch / TensorFlow (choose the one you used)
- ResNet (Pre-trained / custom-trained)
- Streamlit (for web app interface)
- PIL (for image processing)
- scikit-learn (for metrics and evaluation)

## 📊 Results

- **Model Accuracy:** 94%
- **Evaluation Metric:** Accuracy, Precision, Recall, F1-score


## 🖼️ Streamlit Web App

A simple and interactive **Streamlit** interface allows users to upload an image and receive real-time predictions on the type of garbage. The app validates the image, resizes it, and shows the predicted class.

To run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

**📦 Future Improvements**

- ✅ Add **Grad-CAM** or other visualization tools to understand model predictions  
- 🧪 Enhance performance using **data augmentation** and **regularization techniques**  
- 🔁 Experiment with **other architectures** (e.g., EfficientNet, DenseNet) for comparison  
- ☁️ Deploy the model to **Streamlit Cloud**, **Hugging Face Spaces**, or **Docker container**  
- 🌐 Add **multilingual support** to the Streamlit interface for broader accessibility  
- 🗂️ Integrate with a backend database for tracking and storing user uploads and predictions  
- 📱 Build a lightweight mobile/web version of the app for field use in waste sorting  

---

**🙌 Acknowledgments**

- 🤖 Model inspired by the need for **automated waste segregation** to support recycling and sustainability  
- 🗃️ Dataset Source: *[[(https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)]]*  
- 📚 Special thanks to open-source contributors and the **Kaggle** community for insights and feedback  
- 💡 Built using frameworks like **PyTorch/TensorFlow** (choose one), **Streamlit**, and **PIL**



