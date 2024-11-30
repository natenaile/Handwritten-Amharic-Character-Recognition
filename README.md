# Handwritten Amharic Character Recognition through Transfer Learning: Integrating CNN Models and Machine Learning Classifiers.

## Overview

This project focuses on automating the recognition of handwritten **Amharic characters**, which are a part of the Ethiopian script. The Amharic script consists of 238 unique characters, including 34 basic characters with seven variations representing different vowel sounds. Due to the syllabic nature of the script and variations in handwriting styles, recognizing handwritten Amharic characters is a challenging task. In this project, we employ four convolutional neural network (CNN) architectures—**AlexNet**, **VGG16**, **VGG19**, and **ResNet50**—using transfer learning to improve recognition accuracy.

## Methodology

1. **Dataset Overview**:  

- **Total Samples**: 37,752
- **Training Samples**: 30,201
- **Validation Samples**: 7,551
- **Image Size**: 28x28 pixels
- **Image Type**: JPEG

2. **Data Augmentation**:  
   To enhance the robustness and generalization of the models, several **data augmentation** techniques are employed on-the-fly during training, rather than pre-saving new images. These techniques include:
   - **Random Rotation**: The images are randomly rotated within an angle range of ±10 degrees to introduce variability in orientation.
   - **Affine Transformations**: Techniques such as translation (shifting) by 10% of the image dimensions are applied to create slight variations in image positioning.
   - **Random Resized Cropping**: This involves cropping the images from random locations while ensuring that the main part of the character remains visible, and then resizing them back to 224x224 pixels for **VGG16**, **VGG19**, and **ResNet50**, and to 227x227 pixels for **AlexNet**. This approach helps the model learn to focus on different parts of the character images.

3. **Models Used**:
   - **Convolutional Neural Networks (CNNs)**:
     - **AlexNet**
     - **VGG16**
     - **VGG19**
     - **ResNet50**
   - **Alternative Machine Learning Classifiers**:  
     Softmax classifiers are used initially, and they are replaced with **Random Forest**, **XGBoost**, and **Support Vector Machine (SVM)** for comparison.

4. **Transfer Learning**:  
   We leverage pretrained CNN models and fine-tune them for the Amharic character recognition task.

5. **Performance Evaluation**:  
   The models are evaluated based on the following metrics:
   - **Accuracy**  
   - **Precision**  
   - **Recall**  
   - **F1-score**

## Hyperparameters

The following hyperparameters were used for training:

- **Batch Size**: 64
- **Learning Rate**: 0.0001
- **Step Size**: 5
- **Gamma**: 0.3
- **Epochs**: 15

## System Specifications

The experiments were run on the following system:

- **CPU**: Intel i9
- **RAM**: 16 GB
- **GPU**: NVIDIA RTX A2000
- **Storage**: 1 TB HDD
- **Operating System**: Windows 11

### Implementation:
- **Frameworks**: PyTorch, Torchvision

## Results

The system achieved the following performance metrics:

- **Accuracy**: 91.89%
- **Precision**: 92.46%
- **Recall**: 91.89%
- **F1-score**: 91.80%

These results demonstrate the effectiveness of using CNN models in combination with transfer learning for the recognition of complex handwritten Amharic characters. Further research could explore incorporating more advanced techniques to improve recognition performance, especially in the context of varying handwriting styles.

## Key Features

- **High Performance**: The CNN models, especially **ResNet50**, achieved high accuracy in recognizing handwritten Amharic characters.
- **Transfer Learning**: Pretrained models helped to significantly reduce the training time and enhance accuracy.
- **Scalable**: The system can be easily adapted to recognize other scripts or handwriting samples.

## Conclusion

This project showcases the potential of leveraging transfer learning with CNN models for complex handwritten script recognition tasks. The combination of CNNs and machine learning classifiers like Random Forest, XGBoost, and SVM provides a promising approach for improving recognition accuracy and reliability.

## Installation & Usage

### 1. Running the Project in Jupyter Notebook

To run this project in **Jupyter Notebook**, follow these steps:

1. Clone or download this repository to your local machine.
2. Install the necessary dependencies (listed below).
3. Open the notebook in **Jupyter Notebook**.
4. Run the cells sequentially to execute the code and generate the results.

### 2. Dependencies

The notebook requires the following Python libraries:

- **PyTorch**
- **Torchvision**
- **Scikit-learn**
- **Matplotlib**
- **Pandas**
- **NumPy**

You can install these dependencies by running the following command in your terminal or within a Jupyter Notebook cell:

```bash
pip install torch torchvision scikit-learn matplotlib pandas numpy

