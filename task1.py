import cv2
import numpy as np
from common_function import * 


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

        croppedImage = drawBondingRectangleOnImage(im,original)
        
        printTextOnImage("Press S to save",im)

        cv2.imshow("Frame", im)

        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == ord('s'):
            cv2.imwrite('images/task1.png',croppedImage)

    cap.release()
    cv2.destroyAllWindows()

main()