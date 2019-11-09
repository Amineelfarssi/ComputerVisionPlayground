import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import  train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import argparse


ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="output path to the loss/accuracy plot")
args = vars(ap.parse_args())
print("[INFO] loading the full MNIST dataset...")
dataset = fetch_mldata('MNIST original')
data= dataset.data.astype("float")/255.0
(trainX, testX , trainY , testY)= train_test_split(data , dataset.target, test_size=0.25 )

print(trainY.shape)

lb=LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY =lb.transform(testY)

model= Sequential()
model.add(Dense(256,input_shape=(784,),activation="sigmoid"))
model.add(Dense(128,activation="sigmoid"))
model.add(Dense(10,activation="softmax"))


print("[INFO] trainig network ...")

sgd=SGD(0.01)

model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])

H=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,batch_size=128)

print("[INFO] evaluating network ... ")

predictions=model.predict(testX,batch_size=128)

print(classification_report(testY.argmax(axis=1) ,predictions.argmax(axis=1) , target_names= [str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100),H.history['loss'],label="train_loss")
plt.plot(np.arange(0,100),H.history['val_loss'],label="validation_loss")
plt.plot(np.arange(0,100),H.history['accuracy'],label="train_acc")
plt.plot(np.arange(0,100),H.history['val_accuracy'],label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("# Epochs")

plt.ylabel("Loss/accuracy")

plt.legend()

plt.savefig(args["output"])



