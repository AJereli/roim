from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import models
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard

import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# plt.imshow(train_data[1])
# plt.show()

train_data = train_data.reshape((60000, 28, 28, 1))
train_data = train_data.astype('float32') / 255

test_data = test_data.reshape((10000, 28, 28, 1))
test_data = test_data.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='./logs/cnn', histogram_freq=0,
                          write_graph=True, write_images=False)
print(model.summary())
EPOCHS = 30
BATCH_SIZE = 64
history = model.fit(train_data, train_labels,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=[tensorboard, EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=8)])

score = model.evaluate(test_data, test_labels, batch_size=BATCH_SIZE)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])

model.save('mnist_cnn')