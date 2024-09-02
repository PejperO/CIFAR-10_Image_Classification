import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.datasets import cifar10
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

# Definition of class labels
labels = ["Samolot","Samochód","Ptaszek","Kot","Jelonek","Piesek","Żaba","Koń","Statek","Ciężarówka"]

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Preview of the first image in the training set
print("Etykieta: ", labels[y_train[0][0]])
plt.imshow(x_train[0])

# Process data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to categories
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model using Functional API
input_shape=(32,32,3)
inputs = Input(shape=input_shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(inputs)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.4)(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callback settings
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

#Train Model
history = model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1, validation_data=(x_test, y_test),
                    callbacks=[reduce_lr, early_stopping])

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
