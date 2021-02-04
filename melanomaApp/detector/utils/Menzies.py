import cv2
import numpy as np
from .Diameter import Diameter
from .Contours import Contours
from .Color import Color
from .Asymmetry import Asymmetry
from .Preprocess import Preprocess
from .Caracteristics import Caracteristics

class Menzies:
    '''
        the Menzies method
    '''
    def asymmetry(img, contour):
        '''
            returns the asymmetry of the lesion, using homologue method
        '''
        asymmetry = Asymmetry.asymmetryIndex(img, contour)
        return asymmetry
    
    def color(img, contour):
        '''
            returns the number of colors
        '''
        nbColors = Color.colorHSVIntervals(img, contour)
        return nbColors
    
    def darkPoints(img, contour):
        '''
            returns the number of dark points, globules, pigmented network, pseudopods in the lesion
        '''
        # extract the lesion
        lesion = Caracteristics.extractLesion(img, contour)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # crop the rect
        lesion = lesion[y:y + h, x:x + w]
        # mean color of lesion
        mean = np.sum(lesion[lesion[:,:,0]>0])
        mean = mean / np.count_nonzero(lesion[lesion[:,:,0]>0])
        mean = int(mean * 0.9)
        # colors intervals
        blackH = np.array([1, 1, 1], dtype=np.uint8)
        blackL = np.array([mean, mean, mean], dtype=np.uint8)
        brownH1 = np.array([13, 81, 47], dtype=np.uint8)
        brownL1 = np.array([53, 121, 87], dtype=np.uint8)
        brownH2 = np.array([8 ,34, 151], dtype=np.uint8)
        brownL2 = np.array([48, 74, 191], dtype=np.uint8)
        brownH3 = np.array([25 ,21, 70], dtype=np.uint8)
        brownL3 = np.array([65, 61, 110], dtype=np.uint8)
        brownH4 = np.array([24 ,25, 102], dtype=np.uint8)
        brownL4 = np.array([64, 65, 152], dtype=np.uint8)
        intervalsL = [blackH, brownH1, brownH2, brownH3, brownH4]
        intervalsH = [blackL, brownL1, brownL2, brownL3, brownL4]
        res = np.zeros(np.shape(lesion)[:2], dtype=np.uint8)
        for i in range(0, len(intervalsH)):
            L = intervalsL[i]
            H = intervalsH[i]
            mask = cv2.inRange(lesion, L, H)
            res = res + mask
        # get contours
        ret, thresh = cv2.threshold(res, 0, 100, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        c, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # filter contours
        def f(cnt):
            return cv2.arcLength(cnt, closed=True)>10 and cv2.arcLength(cnt, closed=True)<30
        contours = list(filter(f, contours))
        # number of positives
        n = 0
        # lesion area
        lesionArea = cv2.contourArea(contour)
        for cnt in contours:
            n = n + cv2.contourArea(cnt)
        n = n / lesionArea * 100
        n = round(n ,2)
        return n
    
    def blueGrey(img, contour):
        '''
            check for blue grey
        '''
        # extract the lesion
        lesion = Caracteristics.extractLesion(img, contour)
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # crop the rect
        lesion = lesion[y:y + h, x:x + w]
        # set color intervals
        blueGrayH1 = np.array([90, 60, 50], dtype=np.uint8)
        blueGrayL1 = np.array([130, 100, 90], dtype=np.uint8)
        blueGrayH2 = np.array([60, 50, 50], dtype=np.uint8)
        blueGrayL2 = np.array([100, 90, 90], dtype=np.uint8)
        intervalsL = [blueGrayH1, blueGrayH2]
        intervalsH = [blueGrayL1, blueGrayL2]
        res = np.zeros(np.shape(lesion)[:2], dtype=np.uint8)
        for i in range(0, len(intervalsH)):
            L = intervalsL[i]
            H = intervalsH[i]
            mask = cv2.inRange(lesion, L, H)
            res = res + mask
        # blue grey surface area
        s = np.count_nonzero(mask[mask>0]) / lesionArea * 100
        s = round(s, 2)
        return s