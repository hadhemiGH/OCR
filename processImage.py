# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
import cv2
import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True

#parse arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i","--image",required=True,help="Path to image to recognize")
#args = vars(ap.parse_args())
# load and prepare the image
def resize_image(img, size=(28,28)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    if dif > (size[0]+size[1])//2:
        interpolation = cv2.INTER_AREA
    else:
        interpolation=cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.full((dif, dif),255, dtype=img.dtype)
        
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.full((dif, dif, c),255, dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)
def load_image(img):
	# load the image
	#img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = resize_image(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img =  np.array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1,28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255
	return img
 
# load an image and predict the class
def run_example(image):
	img   = load_image(image)
	model = load_model('model.h5')
	digit = model.predict_classes(img)
	#result= model.predict(img)
	#print(result)
	return (digit[0])
