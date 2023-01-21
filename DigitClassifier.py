import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import cv2

#region Neural Network

#loading mnist dataset
mnist = tf.keras.datasets.mnist

#divide into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#check shape
print(x_train.shape)

#normalizing the data for pre processing
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

#resizing images to suit cnn operations
IMG_SIZE = 28
x_trainer = np.array(x_train).reshape(x_train.shape[0], 28, 28, 1)
x_tester = np.array(x_test).reshape(x_test.shape[0], 28, 28, 1)

print("Training Samples Dimension", x_trainer.shape)
print("Testing Samples Dimension", x_tester.shape)

#creating the neural network
model = Sequential()
filter = 64

#first convolution layer   28-3+1 = 26x26
#model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=x_trainer.shape[1:]))
model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same", input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

#second convolution layer  26-3+1 = 24x24
model.add(Conv2D(filter, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

#third convolution layer
model.add(Conv2D(filter, (3, 3), strides=(1, 1), padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

#fully connected layer #1  20x20 = 400
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))

#last fully connected layer #2
model.add(Dense(10))
model.add(Activation("softmax"))

print(model.summary())
print("Total Training samples = ", len(x_trainer))

#options for learning rates
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
model.fit(x_trainer, y_train, batch_size=32, validation_data=(x_tester, y_test), epochs=5)

#Evaluation 
test_loss, test_acc = model.evaluate(x_tester, y_test, batch_size=1)
print("Test Loss on 10,000 test samples: ", test_loss)
print("Validation Accuracy on 10,000 test samples: ", test_acc)

#Predictions
predictions = model.predict([x_tester])
#print(predictions)


#region test to understand predictions
#print(np.argmax(predictions[0]))
#plt.imshow(x_test[0])
#plt.show()
#print(np.argmax(predictions[124]))
#plt.imshow(x_test[124])
#plt.show()
#endregion

#endregion

#region Functions for testing

def predictSamples(file):
    img = cv2.imread('non-mnist/' + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    newimg = tf.keras.utils.normalize(resized, axis=1)
    newimg = newimg.reshape(-1, 28, 28, 1)
    predictions = model.predict(newimg)
    return np.argmax(predictions)


def predictTest(file):
    img = cv2.imread('mnist/' + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    newimg = tf.keras.utils.normalize(resized, axis=1)
    newimg = newimg.reshape(-1, 28, 28, 1)
    predictions = model.predict(newimg)
    return np.argmax(predictions)

#endregion

#region other testing
results = {
    "a": "0",
    "b": "0",
    "c": "0",
    "d": "0",
    "e": "0",
    "f": "0",
    "g": "0",
    "h": "0",
    "i": "0",
    "j": "0"
}
i = 0
j = 0
score = 0
total = 0


def checkScore2(answer, toCheck):
    global score
    if toCheck["a"] == answer:
        score = score + 1
    if toCheck["b"] == answer:
        score = score + 1
    if toCheck["c"] == answer:
        score = score + 1
    if toCheck["d"] == answer:
        score = score + 1
    if toCheck["e"] == answer:
        score = score + 1
    if toCheck["f"] == answer:
        score = score + 1
    if toCheck["g"] == answer:
        score = score + 1
    if toCheck["h"] == answer:
        score = score + 1
    if toCheck["i"] == answer:
        score = score + 1
    if toCheck["j"] == answer:
        score = score + 1
#endregion


score = 0
print("\n")
while i < 10:
    print("Classifying mnist image number: ", i)
    num = str(i)
    results["a"] = predictTest(num + "a.png")
    results["b"] = predictTest(num + "b.png")
    results["c"] = predictTest(num + "c.png")
    results["d"] = predictTest(num + "d.png")
    results["e"] = predictTest(num + "e.png")
    results["f"] = predictTest(num + "f.png")
    results["g"] = predictTest(num + "g.png")
    results["h"] = predictTest(num + "h.png")
    results["i"] = predictTest(num + "i.png")
    results["j"] = predictTest(num + "j.png")
    print(results)
    checkScore2(i, results)
    i = i + 1

print("\nMnist Prediction Result: " + str(score) + "/" + "100")

score = 0
print("\n")
while j < 10:
    print("Classifying non-mnist image number: ", j)
    num = str(j)
    results["a"] = predictSamples(num + "a.png")
    results["b"] = predictSamples(num + "b.png")
    results["c"] = predictSamples(num + "c.png")
    results["d"] = predictSamples(num + "d.png")
    results["e"] = predictSamples(num + "e.png")
    results["f"] = predictSamples(num + "f.png")
    results["g"] = predictSamples(num + "g.png")
    results["h"] = predictSamples(num + "h.png")
    results["i"] = predictSamples(num + "i.png")
    results["j"] = predictSamples(num + "j.png")
    print(results)
    checkScore2(j, results)
    j = j + 1

print("\nNon-Mnist Prediction Result: " + str(score) + "/" + "100")

