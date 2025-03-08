import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# Let's load the dataset using keras
(train_images, train_labels), (test_images,test_labels) = mnist.load_data() # tensorflow has the MNIST dataset built in, so we don't need to get it from a website and can just load it using this function

# Display the shape of the dataset
print(f"Training data shape: {train_images.shape}, Training labels shape: {train_labels.shape}")
print(f"Test data shape: {test_images.shape}, Test labels shape: {test_labels.shape}")

# This is the first time taking a look at the MNIST dataset , so I want to take a good look at what the data looks like.

# Let's look at the first image. The images are just a 28x28 table of numbers between 0 and 255. With this we give each number a grayscale value and then show it.
plt.figure(figsize=(5,5))
plt.imshow(train_images[0], cmap='gray')
plt.colorbar()
plt.title(f"Label: {train_labels[0]}")
plt.show()

# Get min and max values of pixel intensities
min_pixel_value = np.min(train_images) 
max_pixel_value = np.max(train_images) 

print(f"Min pixel value: {min_pixel_value}") # Minimum value is 0, so black = 0
print(f"Max pixel value: {max_pixel_value}") # Max value is 255, so white = 255

# 255 is a big value, so lets normalize the data to a range between 0 and 1. Bigger numbers can get out of hand easily, and take longer to calculate.
if train_images.max() == 255: # if statement, just so I don't accidentally divide the data by 255 twice
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

# And to verify the normalization
print(f"Min pixel value after normalization: {train_images.min()}")
print(f"Max pixel value after normalization: {train_images.max()}")

# The neural net I want to use only accepts vectors, so let's reshape the 2d image into a 1d vector.

# train_images.shape[0] gives the number of images
# -1 tells NumPy to calculate the size of this dimension automatically
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)

# Verify the new shape
print(f"New shape of training images: {train_images_flattened.shape}")
print(f"New shape of test images: {test_images_flattened.shape}")

# Instead of a 28x28 image it is now a list of 784 numbers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# As an initial test NN, we're going to use two hidden layers, the first with 128, and the second with 64. Both using Relu as the activation function.
# This is just meant as a test. We'll try different amounts of laters and neurons later.
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Then to train the model we compiled above
history = model.fit(train_images_flattened, train_labels, 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2) # 20% of the training set is used to validate the model. This is to make sure the score is independent from the training data.

# let's see what the accuracy of the model is using the test data
test_loss, test_accuracy = model.evaluate(test_images_flattened, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# A function to create a model, with as inputs the amount of layers and neurons as hyperparameters.
def create_model(layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(784,), activation='relu'))
    
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameters to try
layers_options = [1, 2, 3]
neurons_options = [128, 64, 32, 16]
epoch = 10

# Variables to store the best configuration
best_accuracy = 0
best_params = {'layers': None, 'neurons': None}

# Perform the grid search
for layers in layers_options:
    for neurons in neurons_options:
        
        model = create_model(layers, neurons)
        
        # Train the model
        history = model.fit(train_images_flattened, train_labels, 
                            epochs=epoch, 
                            batch_size=32, 
                            validation_split=0.2, 
                            verbose=0)
        
        # Get the accuracy on the validation set
        val_accuracy = history.history['val_accuracy'][-1]
    
        print("")
        print(f"Training model with {layers} layers and {neurons} neurons per layer...")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("")
        
        # Check if this is the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params['layers'] = layers
            best_params['neurons'] = neurons

print(f"Best validation accuracy: {best_accuracy:.4f} with layers: {best_params['layers']} and neurons: {best_params['neurons']}")



# Hyperparameters to try
layers_options = [3]
neurons_options = [256, 512, 1024]

# Variables to store the best configuration
best_accuracy = 0
best_params = None

# Perform the grid search
for neurons in neurons_options:
    
    model = create_model(layers_options[0], neurons)
        
    # Train the model
    history = model.fit(train_images_flattened, train_labels, 
                        epochs=epoch, 
                        batch_size=32, 
                        validation_split=0.2, 
                        verbose=0)
        
    # Get the accuracy on the validation set
    val_accuracy = history.history['val_accuracy'][-1]

    print("")
    print(f"Training model with {neurons} neurons...")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print("")
        
    # Check if this is the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_params = neurons

print(f"Best validation accuracy: {best_accuracy:.4f} with neurons: {best_params}")

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# Function to create model with dropout
def create_dropout_model(neurons, dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(784,), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the best model with 256 neurons and dropout
model = create_model(3, 256)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(train_images_flattened, train_labels, 
                    epochs=50,  # Increase the number of epochs. It needs to be more, because we want it to go on until EarlyStopping has a chance to kick in.
                    batch_size=32, 
                    validation_split=0.2, 
                    callbacks=[early_stopping], 
                    verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images_flattened, test_labels, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Function to augment data
def augment_data(images):
    # Original images to [0, 1]
    original_images = images
    
    # Generate negative space features
    negative_images = 1.0 - images
    
    # Generate scaled features to [-1, 1]
    scaled_images = images * 2.0 - 1.0
    
    # Concatenate the original, negative, and scaled features along the second dimension
    augmented_images = np.concatenate([original_images, negative_images, scaled_images], axis=1)
    
    return augmented_images

# Apply the augmentation to training and test datasets
augmented_train_images = augment_data(train_images_flattened)
augmented_test_images = augment_data(test_images_flattened)

# Print the new shape of the augmented datasets
print(f"Augmented train images shape: {augmented_train_images.shape}")
print(f"Augmented test images shape: {augmented_test_images.shape}")

# First I want to try a simple model to see if it even does any better.
# 1 layer with 128 neurons has an accurace of 97.03 %. Let's try that.

# Function to create a simple neural network model
def create_simple_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Get the input shape from the augmented dataset
input_shape = augmented_train_images.shape[1]

# Create the model
model = create_simple_model(input_shape)

# Train the model
history = model.fit(augmented_train_images, train_labels, 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2, 
                    verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(augmented_test_images, test_labels, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Function to augment data
def augment_data_separately(images):
    # Original images to [0, 1]
    original_images = images
    
    # Generate negative space features
    negative_images = 1.0 - images
    
    # Generate scaled features to [-1, 1]
    scaled_images = images * 2.0 - 1.0
    
    return original_images, negative_images, scaled_images

# Apply the augmentation to training and test datasets
train_images_original, train_images_negative, train_images_scaled = augment_data_separately(train_images_flattened)
test_images_original, test_images_negative, test_images_scaled = augment_data_separately(test_images_flattened)

# Print the new shape of the augmented datasets
print(f"Train images (original) shape: {train_images_original.shape}")
print(f"Train images (negative) shape: {train_images_negative.shape}")
print(f"Train images (scaled) shape: {train_images_scaled.shape}")


# Get the input shape from the dataset
input_shape = train_images_original.shape[1]

# Create the models
model_original = create_simple_model(input_shape)
model_negative = create_simple_model(input_shape)
model_scaled = create_simple_model(input_shape)

# Train the models separately
history_original = model_original.fit(train_images_original, train_labels, 
                                          epochs=10, 
                                          batch_size=32, 
                                          validation_split=0.2, 
                                          verbose=1)

history_negative = model_negative.fit(train_images_negative, train_labels, 
                                      epochs=10, 
                                      batch_size=32, 
                                      validation_split=0.2, 
                                      verbose=1)

history_scaled = model_scaled.fit(train_images_scaled, train_labels, 
                                  epochs=10, 
                                  batch_size=32, 
                                  validation_split=0.2, 
                                  verbose=1)

# Evaluate the models on the test set
test_loss_original, test_accuracy_original = model_original.evaluate(test_images_original, test_labels, verbose=0)
test_loss_negative, test_accuracy_negative = model_negative.evaluate(test_images_negative, test_labels, verbose=0)
test_loss_scaled, test_accuracy_scaled = model_scaled.evaluate(test_images_scaled, test_labels, verbose=0)

print(f"Test accuracy (original): {test_accuracy_original:.4f}")
print(f"Test accuracy (negative): {test_accuracy_negative:.4f}")
print(f"Test accuracy (scaled): {test_accuracy_scaled:.4f}")


# Function to create a more complex neural network model
def create_complex_model(input_shape):
    model = Sequential()
    model.add(Dense(512, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Get the input shape from the dataset
input_shape = train_images_original.shape[1]
input_shape_all = augmented_train_images.shape[1]

# Create the models
model_one = create_complex_model(input_shape)
model_all = create_complex_model(input_shape_all)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the models
history_all = model_all.fit(augmented_train_images, train_labels, 
                                  epochs=50, 
                                  batch_size=128, 
                                  validation_split=0.2, 
                                  callbacks=[early_stopping], 
                                  verbose=1)

history_original = model_one.fit(train_images_original, train_labels, 
                                          epochs=50,
                                          batch_size=128, 
                                          validation_split=0.2, 
                                          callbacks=[early_stopping], 
                                          verbose=1)

history_negative = model_one.fit(train_images_negative, train_labels, 
                                      epochs=50, 
                                      batch_size=128, 
                                      validation_split=0.2, 
                                      callbacks=[early_stopping], 
                                      verbose=1)

history_scaled = model_one.fit(train_images_scaled, train_labels, 
                                  epochs=50, 
                                  batch_size=128, 
                                  validation_split=0.2, 
                                  callbacks=[early_stopping], 
                                  verbose=1)

# Evaluate the models on the test set
test_loss_all, test_accuracy_all = model_all.evaluate(augmented_test_images, test_labels, verbose=0)
test_loss_original, test_accuracy_original = model_original.evaluate(test_images_original, test_labels, verbose=0)
test_loss_negative, test_accuracy_negative = model_negative.evaluate(test_images_negative, test_labels, verbose=0)
test_loss_scaled, test_accuracy_scaled = model_scaled.evaluate(test_images_scaled, test_labels, verbose=0)

print(f"Test accuracy (all): {test_accuracy_all:.4f}")
print(f"Test accuracy (original): {test_accuracy_original:.4f}")
print(f"Test accuracy (negative): {test_accuracy_negative:.4f}")
print(f"Test accuracy (scaled): {test_accuracy_scaled:.4f}")
