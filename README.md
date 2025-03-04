# Brain Tumor Classification using CNNs

## Overview
Brain tumor classification is a crucial task in medical imaging, aiding in accurate diagnosis and treatment planning. This project focuses on classifying brain tumors into four categories:
- **Glioma**
- **Meningioma**
- **Healthy**
- **Pituitary**

We employ magnetic resonance imaging (MRI) scans and deep learning techniques to achieve high classification accuracy.

## Dataset
The dataset used in this project consists of MRI scans categorized into the four mentioned classes. The dataset was split into training, validation, and test subsets to ensure robust model evaluation.

## Methodology
### Data Augmentation
To address data scarcity and improve model generalization, data augmentation techniques were applied to all dataset subsets.

### Model Architectures
Several pretrained Convolutional Neural Networks (CNNs) were fine-tuned and evaluated:
- **ResNet-50**
- **VGG16**
- **VGG19**
- **InceptionV3** (Best performing model)
- **MobileNetV2**
- **Custom CNN** (Developed and evaluated for comparison)

Additionally, hyperparameter optimization was performed using **grid search** on MobileNetV2 to identify the best configuration.

## Results
The best-performing models achieved the following classification accuracies on the test set:
- **InceptionV3**: **99.0%** (Best model)
- **VGG16**: 97.6%
- **ResNet-50**: 97.2%
- **MobileNetV2**: 94.6%
- **VGG19**: 93.9%

## Installation
To run this project, follow these steps:

### Prerequisites
Ensure you have the following installed:
- Python (>=3.7)
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

## Acknowledgments
Special thanks to open-source datasets and pretrained model contributors for their support in deep learning research.

## References
Kaggle Dataset: [Brain MRI Images Dataset]([https://www.kaggle.com](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
