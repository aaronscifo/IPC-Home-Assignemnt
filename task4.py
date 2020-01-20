import cv2
import numpy as np
import dlib
from common_function import *
from eye_detection import *
from mouth_detection import *
import numpy as np
import argparse
import imutils
import dlib
import cv2

Detector = None
Predictor = None
Cap = None


CurrentEyeState = 'open'
CurrentMouthState = 'open'


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
		(start, end) = handleFaces(rects, im, gray)

		drawUI(im)
		cv2.imshow("Frame", im)

		key = cv2.waitKey(1)
		if key == 27:  # ESC is pressed
			break
		if key == ord(' '):
			croppedImage = original[start[1]: end[1], start[0]: end[0]]
			cv2.imwrite('images/task3.png', croppedImage)

	Cap.release()
	cv2.destroyAllWindows()


def handleFaces(rects, image, gray):
	global CurrentEyeState
	# loop over the face detections
	xScaled, yScaled, wScaled, hScaled = (0, 0, 0, 0)
	scaledBoundingBoxCoord = ((0, 0), (0, 0))
	for (i, rect) in enumerate(rects):

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = rect_to_bb(rect, 1, 1)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = Predictor(gray, rect)
		shape = shape_to_np(shape)

		# detect eye state
		CurrentEyeState = detectEyeState(shape, image, True)
		# print(CurrentEyeState)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		# for (x, y) in shape:
		# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	return scaledBoundingBoxCoord


def drawUI(frame):
	global CurrentEyeState
	
	cv2.putText(frame, "Eye State: {}".format(CurrentEyeState), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	# cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


init()
main()
