import cv2
import numpy as np
import dlib
from collections import OrderedDict


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


def rect_to_bb(rect, scaleX=1, scaleY=1):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right()  - x
	h = rect.bottom() - y
	print("Original : "+str(x)+' '+str(y)+' '+str(w)+' '+str(h) )

	if(scaleX > 1 or scaleY > 1):
		# scaleX = scaleX/100
		# scaleY = scaleY/100
		wAdd =  round((w * scaleX) /2)
		hAdd =  round((h * scaleY ) /2)#round((scaleY/100 * h )/2  )
		
		w += wAdd
		h += hAdd

		x -= round(wAdd/2)
		y -= round(hAdd/2)

		print("Resized : "+str(x)+' '+str(y)+' '+str(w)+' '+str(h) )

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def resizeImage(img,height,width):
    dim = (width, height)
 
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return resized

#    Conserve aspect ratio of the original region. Useful when shrinking/enlarging
#    images to fit into a certain area.  
#    @param {Number} srcWidth width of source image
#    @param {Number} srcHeight height of source image
#    @param {Number} maxWidth maximum available width
#    @param {Number} maxHeight maximum available height
#    @return {Object} { width, height }
def calculateAspectRatioFit(srcWidth, srcHeight, maxWidth, maxHeight):

    ratio = min(maxWidth / srcWidth, maxHeight / srcHeight)

    return [srcHeight*ratio,srcWidth*ratio]


def printTextOnImage(textStr,image):
    position = (20,25)

    cv2.putText(
        image, #numpy array on which text is written
        textStr, #text
        position, #position at which writing has to start
        cv2.FONT_HERSHEY_PLAIN, #font family
        2, #font size
        (0, 0, 255, 255), #font color
        3
    ) #font stroke

# Draw a blue rectangle on top of an IMG , the orignalImage is a ref to an
# untransformed image as given from
def drawBondingRectangleOnImage(img,originalImage):
    h, w = img.shape[:2]
    ratio = [0.6,0.7]

    ratioScaleRectW,ratioScaleRectH = calculateAspectRatioFit(w, h,ratio[0] * w, ratio[1] * h )

    start_point = ( int( w/2 - ratioScaleRectW/2), int( h/2 - ratioScaleRectH/2 ) )  
    end_point   = ( int(start_point[0] + ratioScaleRectW ), int( start_point[1] + ratioScaleRectH   ) ) 

    boxSize = (ratioScaleRectW, ratioScaleRectH)
    # print('Resolution  ' + str(boxSize[0]) + 'x' + str(boxSize[1]) + ' = ' + str(boxSize[0]/boxSize[1]))

    # Blue color in BGR 
    color = (255, 0, 0) 

    # Line thickness of 2 px 
    thickness = 2
    
    # cropped_img = original.crop((w//2 - 50//2, h//2 - 50//2, w//2 + 50//2, h//2 + 50//2))
    
    imageWithRectangle = cv2.rectangle(img,  start_point, end_point,color,thickness)

    return originalImage[start_point[1]:end_point[1], start_point[0]:end_point[0]]

#https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/


#The dlib face landmark detector will return a shape  object containing the 68 (x, y)-coordinates of the facial landmark regions.
#Using the shape_to_np  function, we cam convert this object to a NumPy array, allowing it to “play nicer” with our Python code.
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords