from scipy.spatial import distance as dist
from common_function import *
import cv2


EYE_AR_THRESH = 0.19


def detectEyeState(shape, image, drawEye=True):
	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

	# extract the left and right eye coordinates, then use the
	# coordinates to compute the eye aspect ratio for both eyes
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]
	leftEAR = eye_aspect_ratio(leftEye)
	rightEAR = eye_aspect_ratio(rightEye)

	# average the eye aspect ratio together for both eyes
	ear = (leftEAR + rightEAR) / 2.0

	# compute the convex hull for the left and right eye, then
	# visualize each of the eyes
	if drawEye == True:
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

	# check to see if the eye aspect ratio is below the EYE_AR_THRESH
	# return the appropriate value based on this descion
	if ear < EYE_AR_THRESH:
		return 'closed'

	return 'open'


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
