# CIFAR-10 Image Classification
This project implements and tests a Convolutional Neural Network (CNN) model for classifying images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 color images in 10 different classes, with each image having a resolution of 32x32 pixels. The goal of this project is to accurately classify these images into their respective categories such as airplanes, automobiles, birds, cats, etc.

## Project Structure
- **Main.py:** A CNN model script which, after training, achieved an accuracy of **79.84%** on the test data.
- **Main_100epoch.py:** Another CNN model script, which runs for up to **100 epochs**. The training stops early if instability is detected, typically reaching around **90.45%** accuracy with a loss of **0.2655** after 57 epochs.
- **training_history.xlsx:** A spreadsheet containing the training history for the models, including accuracy and loss over epochs.

## Model Architecture
The models in this project were built using the following layers:

- **Conv2D:** Convolutional layers to extract features from the input images.
- **MaxPooling2D:** Pooling layers to down-sample the feature maps.
- **Dropout:** Dropout layers to prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Flatten:** Layer to flatten the feature maps into a single vector.
- **Dense:** Fully connected layers to map the features to the output classes.

| Layer (type)	| Output Shape	| Param # |
| ------------- |:-------------:| ------------- |
| InputLayer	| (None, 32, 32, 3)	| 0	|
| Conv2D	| (None, 32, 32, 32)	| 896	|
| MaxPooling2D	| (None, 16, 16, 32)	| 0	|
| Dropout	| (None, 16, 16, 32)	| 0	|
| Conv2D	| (None, 16, 16, 64)	| 18,496	|
| MaxPooling2D	| (None, 8, 8, 64)	| 0	|
| Dropout	| (None, 8, 8, 64)	| 0	|
| Conv2D	| (None, 8, 8, 128)	| 73,856	|
| MaxPooling2D	| (None, 4, 4, 128)	| 0	|
| Dropout	| (None, 4, 4, 128)	| 0	|
| Flatten	| (None, 2048)	| 0	|
| Dense	| (None, 512)	| 1,049,088	|
| Dropout	| (None, 512)	| 0	|
| Dense	| (None, 10)	| 5,130	|
| SUM	 	| | 1,147,466	|


## Key Notes on Model Stability
For models with more than 10 layers, a kernel_initializer was used to enhance the performance and mitigate the vanishing gradient problem. This technique is essential for deep networks to converge and achieve stable training results.

## Results
### Visualization of the Network

The model architectures can be visualized using tools like TensorBoard or by exporting the model summary. Below is a general structure of the models:
- **Input Layer:** 32x32x3
- **Conv2D + MaxPooling2D Layers:** Multiple layers with varying filters
- **Dropout Layers:** Applied after some Conv2D and Dense layers
- **Flatten Layer:** Transition from 2D feature maps to 1D vector
- **Dense Layers:** Fully connected layers leading to the output layer with 10 units (one for each class)

### Confusion Matrix
A confusion matrix is generated to analyze the performance of the model on test data. This matrix highlights the correct and incorrect classifications across all classes.

![Confusion Matrix](https://github.com/user-attachments/assets/3ddadb9a-8d80-4726-a548-1c70be856076)

### Training History
The training history is recorded and plotted to show the model's accuracy and loss over epochs. This can be found in the Historia Trenowania.xlsx file and can be visualized using plotting libraries like Matplotlib.

![Accuracy](https://github.com/user-attachments/assets/7912d96c-2c08-4611-a313-f3211ee63232)
![Loss](https://github.com/user-attachments/assets/60ff1f56-7b6a-41c1-8a49-6fa852b11433)
![Epoch](https://github.com/user-attachments/assets/6aad5f17-eb10-49f9-a2d1-06c0e41aefb1)

### Final Accuracy
- **Main_79.84.py:** Achieved 79.84% accuracy on the test dataset.
- **Main_100epoch.py:** Achieved up to 90.45% accuracy before encountering stability issues.


## How to Run the Project

1. **Dependencies:** Ensure you have the necessary libraries installed, such as TensorFlow, Keras, and Matplotlib.
2. **Training:** Run the provided scripts to train the models. The models will automatically save the best performing weights.
3. **Evaluation:** The models can be evaluated on the CIFAR-10 test dataset to obtain accuracy and confusion matrix.
4. **Model Export:** The models are exported in .keras format for future use.

## What I Learned
Through this project, the following concepts and techniques were reinforced:

- **Building CNN Models:** Understanding the importance of different layers like Conv2D, MaxPooling2D, and Dropout in image classification tasks.
- **Hyperparameter Tuning:** The effect of changing the number of epochs, learning rate, and the use of kernel initializers to combat the vanishing gradient problem.
- **Model Evaluation:** How to use confusion matrices, accuracy scores, and loss curves to assess model performance.
- **Training Stability:** The challenges of deep network stability and strategies to prevent training from failing.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
