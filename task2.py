import cv2
import numpy as np
import dlib  
from common_function import * 

face_cascade = None
cap = None 



def init():
    global cap,face_cascade
    
    # Create the haar cascade  
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # create the landmark predictor  
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")  

     # create the landmark predictor  
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Frame")


def main():
    while True:
        _, original = cap.read(cv2.IMREAD_UNCHANGED)
        im = original.copy() # Store frame in variable

        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        h, w = im.shape[:2]
        # resized = resizeImage(im, h,w)
        # h, w = resized.shape[:2]

        portraitRectangle,croppedImage = drawBondingRectangleOnImage(im,original)
        cv2.imshow("Frame", portraitRectangle)

        key = cv2.waitKey(1)
        if key == 27:#ESC is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

init()
main()
