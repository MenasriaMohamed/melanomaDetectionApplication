import cv2
import numpy as np

class Diameter:
    '''
        all Asymmetry methods
    '''
    @staticmethod
    def diameterMinEnclosingCircle(img, contour):
        '''
            caluclate Diameter by Diameter of Circle around contour
        '''
        # get enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        d = radius * 2
        return d

    @staticmethod
    def diameterOpenCircle(img, contour):
        '''
            caluclate diameter of open circle of the lesion
        '''
        # lesion perimeter
        lesionPerimeter = cv2.arcLength(contour, True)
        # get diameter of circle with same perimeter as lesionPerimeter
        d = lesionPerimeter / np.pi
        d = round(d, 2)
        return d

    @staticmethod
    def diameterLengtheningIndex(img, contour):
        '''
            caluclate Lengthening Index of the lesion
        '''
        # get moments of contour
        M = cv2.moments(contour)
        # moments of inertia
        lamda1 = (M["m20"] + M["m02"] - np.sqrt(np.power(M["m20"] - M["m02"], 2) + 4 * (np.power(M["m11"], 2)))) / 2
        lamda2 = (M["m20"] + M["m02"] + np.sqrt(np.power(M["m20"] - M["m02"], 2) + 4 * (np.power(M["m11"], 2)))) / 2
        li = lamda1 / lamda2 * 100
        li = round(li, 2)
        return li
