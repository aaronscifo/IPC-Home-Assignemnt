import cv2
import numpy as np
import dlib
from common_function import * 
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
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)#
		# imutils.resize(im, width=500)
		
		# detect faces in the grayscale image
		rects = detector(gray, 1)

		# handle detected faces
		(start,end) = handleFaces(rects, im, gray)

		cv2.imshow("Frame", im)

		key = cv2.waitKey(1)
		if key == 27:  # ESC is pressed
			break
		if key == ord('s'):
			croppedImage = original[start[1] : end[1] , start[0] : end[0]  ]
			cv2.imwrite('images/task3.png',croppedImage)

	cap.release()
	cv2.destroyAllWindows()


def handleFaces(rects, image, gray):
	# loop over the face detections
	xScaled, yScaled , wScaled , hScaled = (0,0,0,0) 
	for (i, rect) in enumerate(rects):
	
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = rect_to_bb(rect,1,1)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		(xScaled, yScaled, wScaled, hScaled) =  rect_to_bb(rect,1.5,2)
		scaledBoundingBoxCoord = (xScaled, yScaled), (xScaled + wScaled, yScaled + hScaled)
		cv2.rectangle(image, scaledBoundingBoxCoord[0] ,scaledBoundingBoxCoord[1] , (0, 255, 0), 2)
		
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)


		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		# for (x, y) in shape:
			# cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	return scaledBoundingBoxCoord

init()
main()
