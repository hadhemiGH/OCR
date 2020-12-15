
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255
	return img
 
# load an image and predict the class
def run_example():
	print("load the image")
	img = load_image('test_pic.png')
	print("load model")
	model = load_model('model.h5')
	print("predict the class")
	digit = model.predict_classes(img)
	print(digit[0])
 
print("entry point, run the example")
run_example()
