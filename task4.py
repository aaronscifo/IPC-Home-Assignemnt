import cv2
import numpy as np
import dlib
from common_function import *
from eye_detection import *
from mouth_detection import *
from foucs_detection import *
from head_rotation_detection import *
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time

Detector = None
Predictor = None
Cap = None


CurrentEyeState = 'open'
CurrentMouthState = 'closed'
CurrentImageFocus = 'good'
CurrentHeadState = 'good'


CurrentEyeStateChangeCount = 0
CurrentMouthStateChangeCount = 0

STATE_CHANGE_THRESHOLD = 4


def init():
	global Cap, Detector, Predictor

	# create the landmark predictor
	Cap = cv2.VideoCapture(0)
	cv2.namedWindow("Frame")

	# Create face dector and predictor
	Detector = dlib.get_frontal_face_detector()
	Predictor = dlib.shape_predictor(
		'data/shape_predictor_68_face_landmarks.dat')


def main():
	while True:
		_, original = Cap.read(cv2.IMREAD_UNCHANGED)
		im = original.copy()  # Store frame in variable

		# load the input image, resize it, and convert it to grayscale
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		# imutils.resize(im, width=500)

		# detect faces in the grayscale image
		rects = Detector(gray, 1)

		# handle detected faces
		(start, end) = handleFaces(rects, im, gray, original)
		croppedImage = drawBondingRectangleOnImage(im, original)

		drawUI(im)
		cv2.imshow("Frame", im)


		key = cv2.waitKey(1)
		if key == 27:  # ESC is pressed
			break
		# if key == ord(' '):
		# 	cv2.imwrite('images/task4.png', croppedImage)
		saveCorrectImage(croppedImage)


	Cap.release()
	cv2.destroyAllWindows()


def handleFaces(rects, image, gray, original):
	global CurrentEyeState, CurrentMouthState,CurrentHeadState, CurrentEyeStateChangeCount, CurrentMouthStateChangeCount, STATE_CHANGE_THRESHOLD, CurrentImageFocus
	# loop over the face detections
	xScaled, yScaled, wScaled, hScaled = (0, 0, 0, 0)
	scaledBoundingBoxCoord = ((0, 0), (0, 0))

	if(len(rects) == 0):
		CurrentEyeState   = 'closed'
		CurrentMouthState = 'open'
		CurrentImageFocus = 'bad'
		CurrentHeadState  = 'bad'

	for (i, rect) in enumerate(rects):
		if(i > 0):
			return scaledBoundingBoxCoord

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = Predictor(gray, rect)
		shape = shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = rect_to_bb(rect, 1, 1)
		# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
		face = image[y:y + h, x:x + w]
		# cv2.imwrite('images/face.png', face)

		# show the face number
		# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# detect eye state,a threshold count before chaning the actual state
		newCurrentEyeState = detectEyeState(shape, image, True)
		if(CurrentEyeState != newCurrentEyeState):
			CurrentEyeStateChangeCount += 1
			if(CurrentEyeStateChangeCount >= STATE_CHANGE_THRESHOLD):
				CurrentEyeState = newCurrentEyeState
		else:
			CurrentEyeStateChangeCount = 0

		# detect eye state, we need to pass a threshold count before chaning the actual state
		newCurrentMouthState = detectMouthState(shape, image, True)
		if(CurrentMouthState != newCurrentMouthState):
			CurrentMouthStateChangeCount += 1
			if(CurrentMouthStateChangeCount >= STATE_CHANGE_THRESHOLD):
				CurrentMouthState = newCurrentMouthState
		else:
			CurrentMouthStateChangeCount = 0


		CurrentImageFocus = getFocusQuality(image)
	  
		CurrentHeadState  = detectHeadState(shape,image)

		# print(currentBlur)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		# for (x, y) in shape:
		# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	return scaledBoundingBoxCoord


def getColorByValue(actualValue, validValue):
	if(actualValue == validValue):
		return (0, 255, 0)
	return (0, 0, 255)

def saveCorrectImage(croppedImage):
	if(CurrentHeadState == 'good' 
	and CurrentImageFocus == 'good' 
	and CurrentEyeState == 'open' 
	and CurrentMouthState == 'closed'
	and CurrentImageFocus == 'good'):
		cv2.imwrite('images/task4.png', croppedImage)
		time.sleep(5)
		pass
	pass

def drawUI(frame):
	# global CurrentEyeState, CurrentMouthState
	cv2.putText(frame, "Eye State: {}".format(CurrentEyeState), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, getColorByValue(CurrentEyeState, 'open'), 2)
	
	cv2.putText(frame, "Mouth State: {}".format(CurrentMouthState), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, getColorByValue(CurrentMouthState, 'closed'), 2)
	
	cv2.putText(frame, "Good Focus: {}".format(CurrentImageFocus), (300,  450),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, getColorByValue(CurrentImageFocus, 'good'), 2)

	cv2.putText(frame, "Head State: {}".format(CurrentHeadState), (10, 450),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, getColorByValue(CurrentHeadState, 'good'), 2)

init()
main()
