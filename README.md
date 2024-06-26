# Weather-Classification
A project focused on training computer vision models to classify various weather conditions such as dew, fog/smog, frost, and others.

# Overview
This project develops Convolutional Neural Networks (CNNs) for detecting different weather conditions using images. The models are trained to classify 11 weather types: dew, fog/smog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, and snow. The primary goal is to provide effective tools for weather monitoring and analysis.

# Dataset
The dataset used for this project comprises 6,862 labeled images across the aforementioned 11 weather categories. The data is organized to support training, validation, and testing of machine learning models.

## Sample Images
Here's a glimpse of the dataset:

![Weather Condition Samples](https://github.com/Shreyas1018/Weather-Detection/assets/46682248/c6958fcb-b469-48be-8210-a9dbefbee268)

[Click here to download the dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data).

# Training and Models
The project explores various training approaches, including training models individually, training with combined models, and exploring different architectures.

## Models
Here are the models used in the project:
- **VGG16**
- **ResNet50**
- **Xception**
- **DenseNet121**
- **MobileNetV2**
- **InceptionV3**
- **EfficientNetB0**
- **InceptionResNetV2**

Each of these models offers unique strengths, such as higher accuracy, reduced training times, or more efficient use of resources.

## Approaches to Training
This project explores several methods for training and testing models:

### Models Trained Separately
Individual pretrained models are trained and evaluated to assess performance. This method allows for a detailed analysis of each model's strengths and weaknesses.

### Models Trained Together
This approach combines the outputs from multiple pretrained models to increase accuracy and robustness. Models that can be combined include VGG16, ResNet50, Xception, DenseNet121, and others.

### MLP Approach
A different architecture, using a multi-layer perceptron (MLP), leverages VGG16 as the base model, followed by custom dense layers. This provides an alternative way to process data and offers flexibility in model design.

## Training Process
The training process involves:
- **Data Augmentation**: Horizontal flips, zooms, rotations, and other transformations to improve generalization.
- **Training and Validation Split**: Typically 80/20, with adjustments as needed.
- **Model Compilation**: Models are compiled with the Adam optimizer and categorical cross-entropy loss.
- **Callbacks for Early Stopping and Learning Rate Reduction**: To avoid overfitting and manage training dynamics.

## Model Evaluation and Testing
Models are evaluated on accuracy, loss, and other metrics. The evaluation process includes visualizations, plots, and prediction functions to assess performance on test data.

# Save and Export Models
Models are saved in various formats, including HDF5, SavedModel, and even TFLite for mobile deployment. This flexibility allows models to be restored or converted as needed.

# Requirements
To run this project, ensure you have the following dependencies installed:
- Python (>=3.6)
- TensorFlow
- Keras
- OpenCV
- Scikit-Learn
- Seaborn
- Matplotlib
  
# File Overview
The project comprises six primary Jupyter notebooks, each with a unique purpose and role in the overall workflow.

## `model_extraction.ipynb`
- **Purpose**: This notebook focuses on saving trained models and their components for future use. It also demonstrates converting models to TFLite format for easier deployment.
- **Main Activities**:
  - Saving models in different formats: HDF5, SavedModel, and others.
  - Saving model weights and architecture to JSON.
  - Converting models to TFLite for lightweight deployment.
- **Use Cases**: Ideal for creating backup copies of trained models and preparing models for deployment on mobile or embedded devices.

## `weather-test-cnn-separately.ipynb`
- **Purpose**: This notebook tests the performance of individual CNN models for weather condition classification.
- **Main Activities**:
  - Loading pretrained models (like VGG16, ResNet50, Xception) and compiling them with a custom architecture.
  - Evaluating these models on a test dataset.
  - Displaying prediction results and visualizations for test data.
- **Use Cases**: Useful for assessing the performance of individual models to identify their strengths and weaknesses.

## `weather-test-cnn-together.ipynb`
- **Purpose**: This notebook tests the performance of a model that combines outputs from multiple pretrained CNNs.
- **Main Activities**:
  - Concatenating outputs from multiple pretrained models to create a combined model.
  - Compiling and evaluating the combined model on a test dataset.
  - Visualizing prediction results and model performance.
- **Use Cases**: Suitable for exploring combined models' effectiveness and comparing their performance against individually trained models.

## `weather-train-cnn-separately.ipynb`
- **Purpose**: This notebook focuses on training individual CNN models separately for weather condition classification.
- **Main Activities**:
  - Training and evaluating individual models (VGG16, ResNet50, Xception) on a training dataset.
  - Applying data augmentation techniques to improve generalization.
  - Using callbacks for early stopping and learning rate reduction to manage the training process.
- **Use Cases**: Ideal for exploring the training process for different models and identifying the most effective model.

## `weather-train-cnn-together.ipynb`
- **Purpose**: This notebook focuses on training a combined model that merges outputs from multiple pretrained CNNs.
- **Main Activities**:
  - Concatenating outputs from multiple pretrained models to create a combined model.
  - Training the combined model with data augmentation and callbacks to manage training dynamics.
  - Visualizing performance through plots and prediction functions.
- **Use Cases**: Useful for exploring the benefits of combining multiple pretrained models and training them together.

## `weather-train-mlp.ipynb`
- **Purpose**: This notebook explores a different architecture using a multi-layer perceptron (MLP) with VGG16 as the base model.
- **Main Activities**:
  - Creating a custom MLP-based model with additional dense layers.
  - Training and evaluating the MLP-based model on a dataset.
  - Visualizing training progress through plots and evaluating test accuracy.
- **Use Cases**: Ideal for exploring alternative architectures beyond traditional CNNs.

# License
This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

# Acknowledgements
Thanks to the TensorFlow, Keras, OpenCV, and other open-source communities for providing the tools to make this project possible.

# Contributors
- [DADDIOUAMER Redouane](https://github.com/D-Redouane) (Feature Development)