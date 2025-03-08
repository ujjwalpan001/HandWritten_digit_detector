# Handwritten Digit Recognition with Deep Learning

This project aims to build a deep learning model using Tensorflow to recognize handwritten digits from the MNIST dataset. The MNIST dataset is a widely-used benchmark dataset in machine learning, consisting of 28x28 pixel grayscale images of handwritten digits (0 through 9).

## Project Overview

The project explores various techniques and approaches to improve the accuracy of the model, including:

- Data preprocessing and normalization
- Model architecture experimentation (number of layers, neurons)
- Hyperparameter tuning using grid search
- Regularization techniques (Dropout, Early Stopping)
- Data augmentation

The project is implemented as a Jupyter Notebook, with each step clearly documented and explained.

## Dataset

The MNIST dataset is automatically loaded and split into training and test sets using the `tensorflow.keras.datasets` module.

## Dependencies

The following Python libraries are required to run the code:

- Pandas
- NumPy
- Matplotlib
- Tensorflow
- Scikit-learn

## Usage

1. Clone the repository or download the project files.
2. Install the required dependencies.
3. Open the Jupyter Notebook file (`handwritten_digit_recognition.ipynb`).
4. Run the cells sequentially to execute the code and view the results.

## Results

The final model achieved an accuracy of 97.59% on the test set using a complex architecture with Dropout and Early Stopping. However, simpler models with fewer layers and neurons also achieved respectable accuracy, suggesting diminishing returns for increased model complexity on this particular dataset.

## Conclusion

This project provided hands-on experience in building and experimenting with deep learning models for image classification tasks. While the MNIST dataset is relatively simple, the techniques explored in this project can serve as a foundation for tackling more complex computer vision problems.