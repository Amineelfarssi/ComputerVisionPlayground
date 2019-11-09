from Process import imagetoarraypreprocessor
from Process.SimpleDatasetLoader import SimpleDatasetLoader
from Process.SimplePreprocessor import SimplePreprocessor
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required= True ,help= "Path to imput dataset")
ap.add_argument("-m","--model",required=True,help="path to retrained model")
args=vars(ap.parse_args())

classLabels =["cats","dog","panda"]

print("[INFO] Sampling images ...")

imagePaths = np.array(list(paths.list_images(args["dataset"])))

idx =np.random.randint(0,len(imagePaths),size=(20,))

imagePaths = imagePaths[idx]

sp = SimplePreprocessor(32,32)

iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors = [sp,iap])

(data , labels) = sdl.load(imagePaths)

data = data.astype("float")/255.0

print("[INFO] loading pre-trained network ...")

model = load_model(args["model"])

print("[INFO] predicting ...")

preds = model.predict(data,batch_size=32).argmax(axis=1)

for (i,imagePaths) in enumerate(imagePaths):
    image =cv2.imread(imagePaths)
    cv2.putText(image,"LABEL ={}".format(classLabels[preds[i]]),(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
    cv2.imshow("image",image)
    cv2.waitKey(0)







