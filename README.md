# Emotion Recognition Project

This project aims to build a deep learning model for recognizing emotions from facial expressions in images and video streams. The model is trained on a dataset of labeled facial images representing different emotions such as anger, fear, happiness, neutrality, and sadness.

## Dataset

The project uses a dataset containing images of faces labeled with the corresponding emotion. The dataset is organized into separate folders for each emotion class.

## Model Architecture

The project employs a pre-trained ResNet-50 model as the backbone for feature extraction. The final fully connected layer of the ResNet-50 model is replaced with a new linear layer with an output size equal to the number of emotion classes.

## Training

The model is trained using PyTorch and the provided dataset. The training process includes data augmentation techniques such as random resizing, cropping, flipping, and rotation to improve model performance and generalization. The model is trained using the cross-entropy loss function and the Adam optimizer.

During training, the model's performance is monitored on both the training and validation sets, and the model with the best validation accuracy is saved.

## Evaluation

The trained model's performance is evaluated by plotting the training and validation loss and accuracy curves. Additionally, the model is tested on a set of sample images, and the predicted emotions are displayed along with the corresponding images.

## Real-time Emotion Detection

The project also includes a real-time emotion detection component that uses the trained model to detect and recognize emotions from live video streams captured from a webcam. The detected faces are highlighted with bounding boxes, and the predicted emotions are displayed on the video feed.

## Usage

1. Install the required dependencies listed below.
2. Ensure that the dataset is downloaded and placed in the appropriate directory specified in the code.
3. Run the Jupyter Notebook or Python script to train the model and evaluate its performance.
4. For real-time emotion detection, run the appropriate script or code block with a webcam connected to your system.

## Dependencies

The project relies on the following Python libraries:

- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue.

