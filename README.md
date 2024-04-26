# Weather-Classification
Training a computer vision model to detect different types of weather conditions such as dew, fogsmog, frost etc

# Overview
This project aims to develop a Convolutional Neural Network (CNN) model for detecting different conditions of weather using images. The model utilizes deep learning techniques to analyze visual data and identify patterns associated with wildfire occurrences. The goal is to provide an effective tool for detection and monitoring of weather conditions. The pictures are divided into 11 classes: dew, fog/smog, frost, glaze, hail, lightning , rain, rainbow, rime, sandstorm and snow.

# Dataset
This repository focuses on weather classification using a curated dataset. The dataset contains a collection of weather-related conditions.

The dataset contains labeled 6862 images of different types of weather. The dataset is organized to facilitate the training and evaluation of machine learning models for weather classification tasks. With labeled data indicating different weather conditions (such as fog, dew, rainy, etc.), this dataset serves as a valuable resource for building accurate classifiers.

## Glimpse of the Dataset
![download](https://github.com/Shreyas1018/Weather-Detection/assets/46682248/c6958fcb-b469-48be-8210-a9dbefbee268)

<b>[click here for dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data)</b>

## Requirements
To run this project, ensure you have the following dependencies installed:

- Python (>=3.6)
- TensorFlow
- OpenCV
- Keras
- Scikit-Learn
- Seaborn
- Matplotlib

## Models Used
This project explores several pre-trained models to classify weather conditions:

- VGG16
- ResNet50
- Xception

Each model is trained and evaluated to determine which performs best for this task.

## Results and Evaluation
The training process involves splitting the data into training, validation, and test sets. Data augmentation is applied to improve generalization. Models are evaluated based on accuracy and loss. The following models were trained with the results analyzed:

- **VGG16**: Showed a good balance between performance and training time.
- **ResNet50**: Achieved high accuracy but required more epochs to converge.
- **Xception**: Provided the best performance among the three models in early epochs.

## License
This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

# Acknowledgements
Many thanks to the OpenCV, TensorFlow, Keras and Torch groups and contributors. This project would not have been possible without the existence of high quality, open source machine learning libraries.
I would also like to thank the greater open source community, in which the assortment of concrete examples and code were of great help.
<br>This list is not final, as the project is far from done. Any future acknowledgements will be promptly added.

# Contributors
- [DADDIOUAMER Redouane](https://github.com/D-Redouane) (Added some features)