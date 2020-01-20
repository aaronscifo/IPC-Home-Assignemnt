from scipy.spatial import distance as dist
from common_function import *
import cv2


CENTER_OFFSET_X = 10
CENTER_OFFSET_Y = 20


def detectHeadState(shape, image, drawNose=True):
	# grab the indexes of the facial landmarks for CENTER_OFFSET_Y + hCenterthe nose
	(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["nose"]

	# extract the  nose coordinates, then use the
	# coordinates to compute the nose aspect ratio for both noses
	nose = shape[lStart:lEnd]

	noseCenter = nose[2]
	# print(noseCenter)
	# compute the convex hull for the  nose, then
	# visualize each of the noses
	if drawNose == True:
		COLOR = (0, 255, 0)
		cv2.circle(image, (noseCenter[0], noseCenter[1]), 5, COLOR)
		# noseHull = cv2.convexHull(nose)
		# cv2.drawContours(image, [noseHull], -1, (0, 255, 0), 1)

	h, w = image.shape[:2]
	hCenter = h/2
	wCenter = w/2
	
	CorrectHeadYRotation=False
	CorrectHeadXRotation=False

	# roationMessage = 'Good'

	if noseCenter[0] >= (wCenter - CENTER_OFFSET_X) and noseCenter[0] <= (CENTER_OFFSET_X + wCenter) :
		CorrectHeadXRotation=True
		# print('ok x : '+str(noseCenter[0])+' THRESHOLD: '+str(wCenter - CENTER_OFFSET_X) +' - '+str(CENTER_OFFSET_X + wCenter) )
	else:
		CorrectHeadXRotation=False
		# print('NOT ok x : '+str(noseCenter[0])+' THRESHOLD: '+str(wCenter - CENTER_OFFSET_X) +' - '+str(CENTER_OFFSET_X + wCenter) )

	if noseCenter[1] >= (hCenter - CENTER_OFFSET_Y) and noseCenter[1] <= (CENTER_OFFSET_Y + hCenter) :
		CorrectHeadYRotation = True
		# print('ok y : '+str(noseCenter[1])+' THRESHOLD: '+str(hCenter - CENTER_OFFSET_Y) +' - '+str(CENTER_OFFSET_Y + hCenter) )
	else:
		CorrectHeadYRotation = False
		# print('NOT ok y : '+str(noseCenter[1])+' THRESHOLD: '+str(hCenter - CENTER_OFFSET_Y) +' - '+str(CENTER_OFFSET_Y + hCenter) )

	if(CorrectHeadXRotation == True and CorrectHeadYRotation == True):
		return 'good'
	
	return 'bad'

	# print(noseEAR)
	# check to see if the nose aspect ratio is below the EYE_AR_THRESH
	# return the appropriate value based on this descion
	

