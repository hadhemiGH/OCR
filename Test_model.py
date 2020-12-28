# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
import cv2
import numpy as np
#parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image to recognize")
args = vars(ap.parse_args())
# load and prepare the image

def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1,28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255
	return img
 
# load an image and predict the class
def run_example():
	img = load_image(args['image'])
	model = load_model('smart_cosmos.h5')
	digit = model.predict_classes(img)
	#result= model.predict(img)
	#print(result)
	return (digit[0])
	
print(run_example())
