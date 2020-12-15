# import the necessary packages
import pytesseract
import optparse
import cv2
import os

def get_options():
    #parse cmd arguments
    parser = optparse.OptionParser(usage="usage: %prog [options]", description="options")
    parser.add_option("-i", "--image", action="store",help="path to input image to be OCR'd")
    parser.add_option("-p", "--preprocess", type=str, default="thresh",	help="type of preprocessing to be done") # thresh(threshold) or blur
 
    (options, args) = parser.parse_args()
    return options

class OCR():
    def load_pic(self,file_path):
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def check_pic(self,preprocess, gray):
        if preprocess == "thresh":
           gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		# make a check to see if median blurring should be done to remove
		# noise
        elif  preprocess == "blur":
           gray = cv2.medianBlur(gray, 3)
        return gray

if __name__ == '__main__':
    options = get_options()
    pic_path = options.image
    print(pic_path)
    preprocess=options.preprocess
    ocr=OCR()
    pic=ocr.load_pic(pic_path)
    gray=ocr.check_pic(preprocess,pic)
    # write the grayscale image to disk as a temporary file so we can
	# apply OCR to it
    text = pytesseract.image_to_string(gray)
    print(str(text))
	# show the output images
    cv2.imshow("Image", pic)
    cv2.imshow("Output", gray)
    cv2.waitKey(0)