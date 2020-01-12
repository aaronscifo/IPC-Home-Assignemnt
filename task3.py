import cv2
import numpy as np
import dlib
from common_function import *
from dlib_common_functions import *
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

detector = None
predictor = None
cap = None


def init():
	global cap, detector, predictor

	 # create the landmark predictor
	cap = cv2.VideoCapture(0)
	cv2.namedWindow("Frame")

	# Create face dector and predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(
		'data/shape_predictor_68_face_landmarks.dat')


def main():
	while True:
		_, original = cap.read(cv2.IMREAD_UNCHANGED)
		im = original.copy()  # Store frame in variable

		# load the input image, resize it, and convert it to grayscale
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)#imutils.resize(im, width=500),

		# detect faces in the grayscale image
		rects = detector(gray, 1)

		# handle detected faces
		handleFaces(rects, im,gray)

		cv2.imshow("Frame", im)

		key = cv2.waitKey(1)
		if key == 27:  # ESC is pressed
			break

	cap.release()
	cv2.destroyAllWindows()


def handleFaces(rects, image, gray):
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


init()
main()
