import cv2
import numpy as np
from .Diameter import Diameter
from .Contours import Contours
from .Color import Color
from .Preprocess import Preprocess
from .Caracteristics import Caracteristics

class SevenPointsChecklist:
    '''
        the 7 points Glasgow checklist\n
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3635581/
    '''
    def diameterEvolution(img1, contour1, img2, contour2):
        '''
            returns the evolution of diameter in percentage
        '''
        d1 = Diameter.diameterMinEnclosingCircle(img1, contour1)
        d2 = Diameter.diameterMinEnclosingCircle(img2, contour2)
        evolutionD = (d2 - d1) / d1 * 100
        evolutionD = round(evolutionD, 2)
        return evolutionD
    
    def shapeEvolution(img1, contour1, img2, contour2):
        '''
            returns the evolution of shape, 0 means there's no change in shape
        '''
        evolutionS = cv2.matchShapes(contour1, contour2, 1, 0.0)
        evolutionS = abs(round(evolutionS, 2))
        return evolutionS
    
    def colorEvolution(img1, contour1, img2, contour2):
        '''
            returns the evolution of color
        '''
        nbColors1 = Color.colorHSVIntervals(img1, contour1)
        nbColors2 = Color.colorHSVIntervals(img2, contour2)
        evolutionC = abs(nbColors1 - nbColors2)
        return evolutionC
    
    def diameter(img, contour):
        '''
            returns the diameter
        '''
        diameter = Diameter.diameterMinEnclosingCircle(img, contour)
        diameter = round(diameter, 2)
        return diameter
    
    def inflammationAndBloodness(img, contour):
        '''
            returns the inflammation and bloodness factor (presence of red colors)
        '''
        # remove artifact
        img = Preprocess.removeArtifactYUV(img)
        # extract the lesion
        lesion = Caracteristics.extractLesion(img, contour)
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # crop the rect
        lesion = lesion[y:y + h, x:x + w]
        # convert to HSV
        lesionHSV = cv2.cvtColor(lesion, cv2.COLOR_BGR2HSV)
        # set color intervals
        redH1 = np.array([0, 100, 220], dtype=np.uint8)
        redL1 = np.array([10, 125, 253], dtype=np.uint8)
        redH2 = np.array([160, 130, 100], dtype=np.uint8)
        redL2 = np.array([180, 255, 253], dtype=np.uint8)
        intervalsL = [redH1, redH2]
        intervalsH = [redL1, redL2]
        # check colors
        nbColors = 0
        # seuil ( percentage of colr area compared with the total lesion's area)
        seuil = 1
        for i in range(0, len(intervalsH)):
            L = intervalsL[i]
            H = intervalsH[i]
            mask = cv2.inRange(lesionHSV, L, H)
            n = np.sum(mask != 0) / lesionArea * 100
            if n > seuil:
                nbColors += 1
        return nbColors
    
    def sensibility(img, contour):
        '''
            returns the level of lesion crusting (white color, sensibility of the lesion)
        '''
        # remove artifact
        # img = Preprocess.removeArtifactYUV(img)
        # extract the lesion
        lesion = Caracteristics.extractLesion(img, contour)
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # crop the rect
        lesion = lesion[y:y + h, x:x + w]
        # set color intervals
        whiteH = np.array([170, 170, 170], dtype=np.uint8)
        whiteL = np.array([255, 255, 255], dtype=np.uint8)
        intervalsL = [whiteH]
        intervalsH = [whiteL]
        # check colors
        nbColors = 0
        # seuil ( percentage of colr area compared with the total lesion's area)
        seuil = 0.1
        for i in range(0, len(intervalsH)):
            L = intervalsL[i]
            H = intervalsH[i]
            mask = cv2.inRange(lesion, L, H)
            n = np.sum(mask != 0) / lesionArea * 100
            if n > seuil:
                nbColors += 1
        return nbColors