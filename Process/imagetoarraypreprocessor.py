from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self,dataformat= None):
        self.dataformat=dataformat
    def preprocess(self,image):
        return img_to_array(image , data_format= self.dataformat)

