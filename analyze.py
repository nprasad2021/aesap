import io
import os
import sys
import glob
import base64
import shutil
import json
import requests
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

matching = {'Web_camera':['camera', 'webcam'], 'Headset':['headphones', 'headset'],
			'Backpack':['backpack'], 'Keyboard':['computer keyboard', 'laptop replacement keyboard', 'numeric keypad'],
			'Monitor':['display device', 'computer monitor'], 'Mouse':['mouse'],
			'Notebook':['laptop', 'netbook', 'computer'], 'Printer':['printer'],
			'Smartphone':['mobile phone']}
def create_request(PRECURSOR, output_folder):

	ele = {}
	for k in matching.keys():
		ele[k] = (0,0)
	# Instantiates a client
	client = vision.ImageAnnotatorClient()

	# The name of the image file to annotate
	dirs = os.listdir(PRECURSOR)
	total = 0
	correct = 0

	if os.path.exists(output_folder):
		shutil.rmtree(output_folder)
	os.makedirs(output_folder)

	for d in dirs:
		print('Processing', d)
		if os.path.isdir(PRECURSOR + d) and d != 'Tablet':
			os.chdir(PRECURSOR+d)


			catcorrect = 0
			catall = 0
			requests_list = []

			for file in glob.glob("*.jpg"):
				with open(image_filename, 'rb') as image_file:
					content_json_obj = {
						'content': base64.b64encode(image_file.read()).decode('UTF-8')
						}
				feature_json_obj = [{'type':'LABEL_DETECTION',
					'maxResults': 5,
					}]
				requests_list.append({'features':feature_json_obj, 
					'image':content_json_obj})
			with open(output_folder + d + '.json', 'w') as output_file:
				json.dump({'requests': request_list}, output_file)

def process(precursor, output_dir):
	for d in os.listdir(precursor):
		data = open(output_dir + d + '.json', 'rb').read()
		response = requests.post(url='https://vision.googleapis.com/v1/images:annotate?key=AIzaSyA1ML7fcUJAKYShZyAAWZXQp745oeBOAuw',
    		data=data, headers={'Content-Type': 'application/json'})

'''
'''
def sample(file):
	client = vision.ImageAnnotatorClient()
	with io.open(file, 'rb') as image_file:
		content = image_file.read()
	image = types.Image(content=content)
	label_response = client.label_detection(image=image)
	labels = label_response.label_annotations
	for label in labels:
		print(str(label.description))
		if str(label.description) in matching['Keyboard']:
			print('MATCHING')

if __name__ == '__main__':
	PRECURSOR = '/Users/i869533/Documents/image_purchasing/data/' + sys.argv[1] + '/train/'
	OUTPUT_DIR = '/Users/i869533/Documents/image_purchasing/data/' + sys.argv[1] + '/'
	
	create_request(PRECURSOR, OUTPUT_DIR)
	process(PRECURSOR, OUTPUT_DIR)
	#sample('/Users/i869533/Documents/aesap/data/sap_data/train/Keyboard/628111_2.jpg')
