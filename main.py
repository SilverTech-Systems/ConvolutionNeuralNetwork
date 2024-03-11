import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from sklearn.utils import compute_class_weight
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import save_model, load_model

main_path = './Fruit'

image_size = (64, 64)
batch_size = 40

from keras.utils import image_dataset_from_directory

Xtrain = image_dataset_from_directory(main_path,
                                      subset='training',
                                      validation_split=0.2,
                                      image_size=image_size,
                                      batch_size=batch_size,
                                      seed=123)

Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=image_size,
                                    batch_size=batch_size,
                                    seed=123)

classes = Xtrain.class_names
print(classes)

"""---------------------------------------------------------"""
labels = np.array([], dtype=int)

for img, lab in Xtrain:
    labels = np.concatenate((labels, lab.numpy()))
for img, lab in Xval:
    labels = np.concatenate((labels, lab.numpy()))

plt.figure()
plt.title('Broj primeraka po klasama:')
plt.hist(labels, align='left', bins=[0, 1, 2, 3], rwidth=0.5)
plt.xticks([0, 1, 2], classes)
"plt.show()"


"""---------------------------------------------------------"""
N = 3

plt.figure()
plt.title('Primeri podataka')
shown = [False]*3
for img, lab in Xtrain.take(1):
    for i in range(len(img)):
        if not shown[lab[i]]:
            shown[lab[i]] = True
            plt.subplot(1, N, int(lab[i])+1)
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.title(classes[lab[i]])
            plt.axis('off')

"""---------------------------------------------------------"""

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(image_size[0],
                                                 image_size[1], 3)),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.1),
  ]
)
plt.figure()
plt.title('Primjeri efekta slojeva za augmentaciju podataka')
for img, val in Xtrain.take(1):
    for i in range(10):
        newImg = data_augmentation(img)
        plt.subplot(2, 5, i + 1)
        plt.imshow(newImg[0].numpy().astype('uint8'))
        plt.axis('off')

plt.show()

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(4, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(N, activation='softmax')
])


model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

"""---------------------------------------------------------"""
history = model.fit(Xtrain,
                        epochs=30,
                        validation_data=Xval,
                        verbose=1
                    )

"""---------------------------------------------------------"""
trainingAcc = history.history['accuracy']
validationAcc = history.history['val_accuracy']

trainingLoss = history.history['loss']
validationLoss = history.history['val_loss']

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(trainingAcc)
plt.plot(validationAcc)
plt.title('Tačnost')
plt.legend(['Trening skup', 'Validacioni skup'])
plt.subplot(1, 2, 2)
plt.plot(trainingLoss)
plt.plot(validationLoss)
plt.title('Kriterijumska funkcija')
plt.legend(['Trening skup', 'Validacioni skup'])
plt.show()

"""---------------------------------------------------------"""

labelsTrainingData = np.array([])
predictionsTrainingData = np.array([])
for img, lab in Xtrain:
    labelsTrainingData = np.concatenate((labelsTrainingData, lab))
    predictionsTrainingData = np.concatenate((predictionsTrainingData, np.argmax(model.predict(img, verbose=0), axis=1)))

print('Tacnost modela na trening skupu je: '+ str(100*accuracy_score(y_true=labelsTrainingData, y_pred=predictionsTrainingData))+'%')


cmTraining = confusion_matrix(y_true=labelsTrainingData, y_pred=predictionsTrainingData)

total_samples_per_class = np.sum(cmTraining, axis=1, keepdims=True)

cmTrainingPercent = (cmTraining.astype('float') / total_samples_per_class) * 100

cmTrainingDisplay = ConfusionMatrixDisplay(confusion_matrix=cmTrainingPercent, display_labels=classes)
cmTrainingDisplay.plot()
plt.title("Matrica konfuzije na trening skupu[%]")
plt.show()

"""---------------------------------------------------------"""


labelsTest = np.array([])
predTest = np.array([])
correctPredictions = {}
incorrectPredictions = {}
for img, lab in Xval:
    currentPred = np.argmax(model.predict(img, verbose=0), axis=1)
    labelsTest = np.concatenate((labelsTest, lab))
    predTest = np.concatenate((predTest, currentPred))
    for j in range(len(currentPred)):
        if int(lab[j]) == int(currentPred[j]) and correctPredictions.get(int(lab[j])) is None:
            correctPredictions[int(lab[j])] = (int(currentPred[j]), img[j])
        elif int(lab[j]) != int(currentPred[j]) and incorrectPredictions.get(int(lab[j])) is None:
            incorrectPredictions[int(lab[j])] = (int(currentPred[j]), img[j])

plt.figure()
plt.title('Ispravno klasifikovani podaci validacionog skupa')
for i in range(N):
    if correctPredictions.get(i):
        plt.subplot(1, N, i + 1)
        plt.title(classes[correctPredictions[i][0]])
        plt.imshow(correctPredictions[i][1].numpy().astype('uint8'))
        plt.axis('off')

plt.figure()
plt.title('Neispravno klasifikovani podaci validacionog skupa (slika+predikcija)')
for i in range(N):
    if incorrectPredictions.get(i):
        plt.subplot(1, 3, i + 1)
        plt.title(classes[incorrectPredictions[i][0]])
        plt.imshow(incorrectPredictions[i][1].numpy().astype('uint8'))
        plt.axis('off')


print('Tačnost modela na skupu za validaciju je: ' + str(100*accuracy_score(labelsTest, predTest)) + '%')

cm = confusion_matrix(labelsTest, predTest)

total_samples_per_class = np.sum(cm, axis=1, keepdims=True)

cmTrainingPercent = (cm.astype('float') / total_samples_per_class) * 100

cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cmTrainingPercent, display_labels=classes)
cmDisplay.plot()
plt.title("Matrica konfuzije na validacionom skupu[%]")
plt.show()

