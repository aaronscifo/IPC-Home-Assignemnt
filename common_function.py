import cv2
import numpy as np
import dlib  


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
        3) #font stroke

def drawBondingRectangleOnImage(img,originalImage):
    h, w = img.shape[:2]
    ratio = [0.6,0.7]

    ratioScaleRectW,ratioScaleRectH = calculateAspectRatioFit(w, h,ratio[0] * w, ratio[1] * h )

    start_point = ( int( w/2 - ratioScaleRectW/2), int(h/2 - ratioScaleRectH/2) )  
    end_point   = ( int(start_point[0] + ratioScaleRectW ), int(start_point[1] + ratioScaleRectH   ) ) 

    boxSize = (ratioScaleRectW, ratioScaleRectH)
    # print('Resolution  ' + str(boxSize[0]) + 'x' + str(boxSize[1]) + ' = ' + str(boxSize[0]/boxSize[1]))

    # Blue color in BGR 
    color = (255, 0, 0) 

    # Line thickness of 2 px 
    thickness = 2
    
    # cropped_img = original.crop((w//2 - 50//2, h//2 - 50//2, w//2 + 50//2, h//2 + 50//2))
    
    imageWithRectangle = cv2.rectangle(img,  start_point, end_point,color,thickness)

    return originalImage[start_point[1]:end_point[1], start_point[0]:end_point[0]]
