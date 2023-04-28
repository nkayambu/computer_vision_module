#import the necessary packages
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from detection_helpers import sliding_window
from detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow as tf
from PIL import Image

# construct the argument parse and parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)",
	help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1,
	help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())'''

# initialize variables used for the object detection procedure
WIDTH = 600
PYR_SCALE = 2.5
WIN_STEP = 8
ROI_SIZE = (150,175)
INPUT_SIZE = (256, 256)

#load model
model = tf.keras.models.load_model("7_class.model")

#load live video
video = cv2.VideoCapture(0)
#time.sleep(5)

while True:
		time.sleep(5)
		ret, frame = video.read()
		#if (ret == False):
			#print("No camera")
			#break

		im = Image.fromarray(frame)

		# resize for input to model 256x256
		im = im.resize((256, 256))
		img_array = np.array(im)
		# change input to input tensor
		img_array = np.expand_dims(img_array, axis=0)
		#cv2.imshow(captures)

		# grab its dimensions
		(H, W) = img_array[0].shape[:2]

		# initialize the image pyramid
		pyramid = image_pyramid(img_array[0], scale=PYR_SCALE, minSize=ROI_SIZE)
		# initialize two lists, one to hold the ROIs generated from the image
		# pyramid and sliding window, and another list used to store the
		# (x, y)-coordinates of where the ROI was in the original image
		rois = []
		locs = []
		# time how long it takes to loop over the image pyramid layers and
		# sliding window locations
		start = time.time()
		# loop over the image pyramid
		for image in pyramid:
			# determine the scale factor between the *original* image
			# dimensions and the *current* layer of the pyramid
			scale = W / float(image.shape[1])
			# for each layer of the image pyramid, loop over the sliding
			# window locations
			for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
				# scale the (x, y)-coordinates of the ROI with respect to the
				# *original* image dimensions
				x = int(x * scale)
				y = int(y * scale)
				w = int(ROI_SIZE[0] * scale)
				h = int(ROI_SIZE[1] * scale)
				# take the ROI and preprocess it so we can later classify
				# the region using Keras/TensorFlow
				roi = cv2.resize(roiOrig, INPUT_SIZE)
				roi = img_to_array(roi)
				roi = preprocess_input(roi)
				# update our list of ROIs and associated coordinates
				rois.append(roi)
				locs.append((x, y, x + w, y + h))
				# check to see if we are visualizing each of the sliding
				# windows in the image pyramid

		# show how long it took to loop over the image pyramid layers and
		# sliding window locations
		end = time.time()
		print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
			end - start))
		# convert the ROIs to a NumPy array
		rois = np.array(rois, dtype="float32")
		# classify each of the proposal ROIs using CNN and then show how
		# long the classifications took
		print("[INFO] classifying ROIs...")
		start = time.time()
		prediction = model.predict(rois)#
		end = time.time()
		print("[INFO] classifying ROIs took {:.5f} seconds".format(
			end - start))
		prediction = prediction[:, 0:6]
		#print(prediction)

		# decode the predictions and initialize a dictionary which maps class
		# labels (keys) to any ROIs associated with that label (values)
		preds = np.argmax(prediction, axis=1)

		#print(preds)
		#print(len(prediction))
		print(f"class: {preds[0]}, score: {prediction[0][preds[0]]}")
		labels = ["10 speed limit sign", "15 speed limit sign", "25 speed limit sign", "30 speed limit sign", "35 speed limit sign", "45 speed limit sign", "Stop Sign"]

		labels2 = {}
		preds2 = []
		for x in range(len(preds)):
			preds2.append((preds[x], labels[preds[x]], prediction[x][preds[x]]))
			# print(preds2[-1])

		# loop over the predictions
		print(preds2)

		for (i, p) in enumerate(preds2):

			# grab the prediction information for the current ROI
			(imageID, label, prob) = p
			# filter out weak detections by ensuring the predicted probability
			# is greater than the minimum probability
			if prob >= .99:
				#print(p)


				# grab the bounding box associated with the prediction and
				# convert the coordinates
				box = locs[i]
				# grab the list of predictions for the label and add the
				# bounding box and probability to the list
				L = labels2.get(label, [])
				L.append((box, prob))
				labels2[label] = L

		# loop over the labels for each of detected objects in the image
		for label in labels2.keys():
			# clone the original image so that we can draw on it
			#print(f"[INFO] showing results for '{label}'")
			#clone = image_array[0].copy()
			# loop over all bounding boxes for the current label
			'''for (box, prob) in labels2[label]:
				# draw the bounding box on the image
				(startX, startY, endX, endY) = box
				cv2.rectangle(img_array[0], (startX, startY), (endX, endY),
					(0, 255, 0), 2)'''
			# applying non-maxima suppression
			#clone = img_array[0].copy()
			# extract the bounding boxes and associated prediction
			# probabilities, then apply non-maxima suppression
			boxes = np.array([p[0] for p in labels2[label]])
			proba = np.array([p[1] for p in labels2[label]])
			boxes = non_max_suppression(boxes, proba)
			#print("boxes: ", boxes)
			#print(f"prob: {proba}")

			probability = prediction[0][preds[0]]
			probability = "{:.2f}".format(probability)
			text = str(label) + " " + str(probability)
			# loop over all bounding boxes that were kept after applying
			# non-maxima suppression
			for (startX, startY, endX, endY) in boxes:
				# draw the bounding box and label on the image
				cv2.rectangle(img_array[0], (startX, startY), (endX, endY),
							  (0, 255, 0), 2)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.putText(img_array[0], text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
			# show the output after apply non-maxima suppression
			orig = imutils.resize(img_array[0], width=400)
			cv2.imshow("Capturing", orig)
			# key = cv2.waitKey(1)
			# if key == ord('q'):
			# 	break

video.release()
cv2.destroyAllWindows()