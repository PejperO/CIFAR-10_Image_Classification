import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import plot_model
from keras.datasets import cifar10
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Definition of class labels
labels = ["Samolot", "Samochód", "Ptaszek", "Kot", "Jeleń", "Pies", "Żaba", "Koń", "Statek", "Ciężarówka"]

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preview of the first image in the training set
print("Etykieta: ", labels[y_train[0][0]])
plt.imshow(x_train[0])
plt.show()

# Process data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to categories
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model using Functional API
input_shape = (32, 32, 3)
inputs = Input(shape=input_shape)

x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train Model
history = model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=1, validation_data=(x_test, y_test))

# Creating a chart with history
df = pd.DataFrame(history.history)
ax = df.plot()
# Saving the chart to a file
fig = ax.get_figure()
fig.savefig('history_plot.png')

# Model Rating
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Model visualization
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# export CNN model
model.save("my_model_cifar10.keras")

# Displaying examples of misclassified
predictions = np.argmax(model.predict(x_test), axis=1)
y_test_flat = np.argmax(y_test, axis=1)
incorrect_indices = np.nonzero(predictions != y_test_flat)[0]

for i in range(5):
    idx = incorrect_indices[i]
    print("Przykład źle sklasyfikowany nr", i+1)
    plt.imshow(x_test[idx])
    plt.xlabel(f"True label: {labels[y_test_flat[idx]]}, Predicted label: {labels[predictions[idx]]}")
    plt.show()

# Creating a confusion matrix
cm = confusion_matrix(y_test_flat, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

#Creating a chart
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')

# Saving the chart to a file
plt.savefig('confusion_matrix.png')
plt.show()
