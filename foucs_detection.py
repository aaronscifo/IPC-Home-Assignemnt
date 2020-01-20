import cv2

Threshold = 100.0

# compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian


def variance_of_laplacian(image):

    return cv2.Laplacian(image, cv2.CV_64F).var()


def getFocusQuality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < Threshold:
        return "Bad"

    return "Good"
