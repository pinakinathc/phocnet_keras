'''This code loads the word images and it's transcripts.
After this the code will convert the transcript of the word into
PHOC label and feed that PHOC label as the target value of that image

Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>
'''

from glob import glob
import cv2
from xml.etree import ElementTree as ET
import numpy as np

from create_phoc_label import generate_label

# Change the below code
# Enter the directory of your word image and their xml files
WORD_IMAGE_DIR = 'datasets/words/'
XML_DIR = 'datasets/xml/'

def load_data():
	xml_files = glob(XML_DIR+'*.xml')
	xml_files = xml_files[:10000]
	
	x_train = []
	y_train = []
	x_test = []
	y_test = []

	for xml_file in xml_files:
		tree = ET.parse(xml_file)
		root = tree.getroot()

		# Change this code to go to your image directory
		image_dir = xml_file.split('/')[-1].split('.')[0].split('-')
		image_dir = image_dir[0] + '/' +image_dir[0]+'-'+image_dir[1]+'/'
		image_dir = WORD_IMAGE_DIR + image_dir

		for word in root.iter('word'):
			img = cv2.imread(image_dir+word.get('id')+'.png', 0)
			try:
				img = cv2.resize(img, (28, 28))
				img = img[:, :, np.newaxis]
			except:
				continue
			transcript = word.get('text').lower()
			target = generate_label(transcript)
			if sum(target) == 0: # To ignore special characters
				continue

			if np.random.random() <= .3: # To divide into training and test set
				x_test.append(img)
				y_test.append(target)
			else:
				x_train.append(img)
				y_train.append(target)

	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_test = np.array(x_test)
	y_test = np.array(y_test)

	return x_train, y_train, x_test, y_test
