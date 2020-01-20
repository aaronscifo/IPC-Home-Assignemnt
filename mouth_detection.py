from scipy.spatial import distance as dist
from common_function import *
import cv2


MOUTH_AR_THRESH = 0.9


def detectMouthState(shape, image, drawMouth=True):
	# grab the indexes of the facial landmarks for the mouth
	(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["mouth"]

	# extract the  mouth coordinates, then use the
	# coordinates to compute the mouth aspect ratio for both mouths
	mouth = shape[lStart:lEnd]
	mouthEAR = mouth_aspect_ratio(mouth)

	# compute the convex hull for the  mouth, then
	# visualize each of the mouths
	if drawMouth == True:
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(image, [mouthHull], -1, (0, 255, 0), 1)

	# print(mouthEAR)
	# check to see if the mouth aspect ratio is below the EYE_AR_THRESH
	# return the appropriate value based on this descion
	if mouthEAR > MOUTH_AR_THRESH:
		return 'closed'

	return 'open'


def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	vertial = dist.euclidean(mouth[1], mouth[2])
	vertial += dist.euclidean(mouth[2], mouth[3])
	vertial += dist.euclidean(mouth[4], mouth[5])

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	horizontal = dist.euclidean(mouth[6], mouth[7])
	# horizontal = dist.euclidean(mouth[13], mouth[14])
	horizontal = dist.euclidean(mouth[0], mouth[11])

	# compute the mouth aspect ratio
	ear = vertial / (2.0 * horizontal)

	# return the mouth aspect ratio
	return ear
