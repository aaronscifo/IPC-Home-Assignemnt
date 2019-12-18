import cv2
import numpy as np


def nothing(x):
    print(str(x))
    pass


cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame")


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)


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

def getPortraitRectangle(img):
    h, w = img.shape[:2]
    ratio = [0.6,0.7]

    ratioScaleRectW,ratioScaleRectH = calculateAspectRatioFit(w, h,ratio[0] * w, ratio[1] * h )

    start_point = ( int( w/2 - ratioScaleRectW/2), int(h/2 - ratioScaleRectH/2) )  
    end_point   = ( int(start_point[0] + ratioScaleRectW ), int(start_point[1] + ratioScaleRectH   ) ) 

    boxSize = (ratioScaleRectW, ratioScaleRectH)
    print('Resolution  ' + str(boxSize[0]) + 'x' + str(boxSize[1]) + ' = ' + str(boxSize[0]/boxSize[1]))

    # Blue color in BGR 
    color = (255, 0, 0) 

    # Line thickness of 2 px 
    thickness = 2
    
    # cropped_img = original.crop((w//2 - 50//2, h//2 - 50//2, w//2 + 50//2, h//2 + 50//2))
    
    img_mod = cv2.rectangle(img,  start_point, end_point,color,thickness)

    return img_mod

def main():

    while True:
        _, original = cap.read(cv2.IMREAD_UNCHANGED)
        im = original.copy() # Store frame in variable
        
        
        
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        h, w = im.shape[:2]
        # resized = resizeImage(im, h,w)
        # h, w = resized.shape[:2]

        portraitRectangle = getPortraitRectangle(im)
        cv2.imshow("Frame", portraitRectangle)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

main()