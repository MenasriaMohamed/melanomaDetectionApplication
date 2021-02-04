import cv2
import numpy as np
from .Preprocess import Preprocess
from .Caracteristics import Caracteristics

class Color:
    '''
        all Color methods
    '''
    def colorHSVIntervals(img, contour):
        '''
            returns the number of colors in a lesion by assigning colors to HSV intervals
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
        whiteH = np.array([0, 0, 254], dtype=np.uint8)
        whiteL = np.array([180, 250, 255], dtype=np.uint8)
        blackH = np.array([0, 0, 1], dtype=np.uint8)
        blackL = np.array([180, 120, 150], dtype=np.uint8)
        redH1 = np.array([0, 100, 220], dtype=np.uint8)
        redL1 = np.array([10, 125, 253], dtype=np.uint8)
        redH2 = np.array([160, 130, 100], dtype=np.uint8)
        redL2 = np.array([180, 255, 253], dtype=np.uint8)
        darkBrownH1 = np.array([0, 30, 140], dtype=np.uint8)
        darkBrownL1 = np.array([10, 120, 253], dtype=np.uint8)
        darkBrownH2 = np.array([0, 130, 120], dtype=np.uint8)
        darkBrownL2 = np.array([10, 255, 253], dtype=np.uint8)
        lightBrownH1 = np.array([11, 50, 140], dtype=np.uint8)
        lightBrownL1 = np.array([21, 255, 253], dtype=np.uint8)
        lightBrownH2 = np.array([160, 30, 170], dtype=np.uint8)
        lightBrownL2 = np.array([180, 100, 253], dtype=np.uint8)
        lightBrownH3 = np.array([11, 120, 100], dtype=np.uint8)
        lightBrownL3 = np.array([21, 255, 253], dtype=np.uint8)
        blueGrayH1 = np.array([120, 10, 150], dtype=np.uint8)
        blueGrayL1 = np.array([180, 120, 170], dtype=np.uint8)
        blueGrayH2 = np.array([0, 120, 100], dtype=np.uint8)
        blueGrayL2 = np.array([10, 130, 190], dtype=np.uint8)
        intervalsL = [whiteH, blackH, redH1, redH2, darkBrownH1, darkBrownH2, lightBrownH1, lightBrownH2, lightBrownH3, blueGrayH1, blueGrayH2]
        intervalsH = [whiteL, blackL, redL1, redL2, darkBrownL1, darkBrownL2, lightBrownL1, lightBrownL2, lightBrownL3, blueGrayL1, blueGrayL2]
        # check colors
        nbColors = 0
        # seuil ( percentage of colr area compared with the total lesion's area)
        seuil = 6
        for i in range(0, len(intervalsH)-1):
            L = intervalsL[i]
            H = intervalsH[i]
            mask = cv2.inRange(lesionHSV, L, H)
            n = np.sum(mask != 0) / lesionArea * 100
            if n > seuil:
                nbColors += 1
        return nbColors
    
    def colorYUVIntervals(img, contour):
        '''
            returns the number of colors in a lesion by assigning colors to YUV intervals
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
        # convert to YUV
        lesionLUV = cv2.cvtColor(lesion, cv2.COLOR_BGR2YUV)
        # set color intervals
        whiteH = np.array([190, 110, 160], dtype=np.uint8)
        whiteL = np.array([240, 155, 180], dtype=np.uint8)
        blackH = np.array([70, 100, 130], dtype=np.uint8)
        blackL = np.array([110, 120, 139], dtype=np.uint8)
        redH = np.array([175, 130, 160], dtype=np.uint8)
        redL = np.array([180, 160, 200], dtype=np.uint8)
        darkBrownH1 = np.array([30, 100, 140], dtype=np.uint8)
        darkBrownL1 = np.array([60, 120, 155], dtype=np.uint8)
        darkBrownH2 = np.array([60, 105, 140], dtype=np.uint8)
        darkBrownL2 = np.array([220, 140, 155], dtype=np.uint8)
        lightBrownH1 = np.array([70, 115, 160], dtype=np.uint8)
        lightBrownL1 = np.array([175, 180, 190], dtype=np.uint8)
        lightBrownH2 = np.array([180, 120, 170], dtype=np.uint8)
        lightBrownL2 = np.array([220, 145, 190], dtype=np.uint8)
        blueGrayH = np.array([90, 100, 155], dtype=np.uint8)
        blueGrayL = np.array([170, 150, 159], dtype=np.uint8)
        intervalsL = [whiteH, blackH, redH, darkBrownH1, darkBrownH2, lightBrownH1, lightBrownH2, blueGrayH]
        intervalsH = [whiteL, blackL, redL, darkBrownL1, darkBrownL2, lightBrownL1, lightBrownL2, blueGrayL]
        # check colors
        nbColors = 0
        # seuil ( percentage of colr area compared with the total lesion's area)
        seuil = .016
        for i in range(0, len(intervalsH)-1):
            L = intervalsL[i]
            H = intervalsH[i]
            mask = cv2.inRange(lesionLUV, L, H)
            n = np.sum(mask != 0) / lesionArea * 100
            if n > seuil:
                nbColors += 1
        return nbColors
    
    def colorYCbCrIntervals(img, contour):
        '''
            returns the number of colors in a lesion by assigning colors to YCbCr intervals
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
        # convert to YCbCr
        lesionYCbCr = cv2.cvtColor(lesion, cv2.COLOR_BGR2YCrCb)
        # set color intervals
        whiteH = np.array([155, 160, 112], dtype=np.uint8)
        whiteL = np.array([175, 180, 124], dtype=np.uint8)
        blackH = np.array([69, 132, 120], dtype=np.uint8)
        blackL = np.array([115, 141, 129], dtype=np.uint8)
        redH = np.array([81, 127, 126], dtype=np.uint8)
        redL = np.array([109, 133, 132], dtype=np.uint8)
        darkBrownH1 = np.array([70, 142, 116], dtype=np.uint8)
        darkBrownL1 = np.array([170, 168, 124], dtype=np.uint8)
        darkBrownH2 = np.array([37, 138, 107], dtype=np.uint8)
        darkBrownL2 = np.array([67, 153, 123], dtype=np.uint8)
        darkBrownH3 = np.array([139, 142, 110], dtype=np.uint8)
        darkBrownL3 = np.array([170, 168, 124], dtype=np.uint8)
        lightBrownH1 = np.array([64, 147, 67], dtype=np.uint8)
        lightBrownL1 = np.array([205, 192, 109], dtype=np.uint8)
        lightBrownH2 = np.array([110, 155, 90], dtype=np.uint8)
        lightBrownL2 = np.array([180, 165, 105], dtype=np.uint8)
        blueGrayH = np.array([70, 142, 111], dtype=np.uint8)
        blueGrayL = np.array([138, 168, 115], dtype=np.uint8)
        intervalsL = [whiteH, blackH, redH, darkBrownH1, darkBrownH2, darkBrownH3, lightBrownH1, lightBrownH2, blueGrayH]
        intervalsH = [whiteL, blackL, redL, darkBrownL1, darkBrownL2, darkBrownL3, lightBrownL1, lightBrownL2, blueGrayL]
        # check colors
        nbColors = 0
        # seuil ( percentage of colr area compared with the total lesion's area)
        seuil = .08
        for i in range(0, len(intervalsH)-1):
            L = intervalsL[i]
            H = intervalsH[i]
            mask = cv2.inRange(lesionYCbCr, L, H)
            n = np.sum(mask != 0) / lesionArea * 100
            if n > seuil:
                nbColors += 1
        return nbColors
    
    def colorSDG(img, contour):
        '''
            calculate the Standard Deviation Grayscale
        '''
        # remove artifact
        img = Preprocess.removeArtifactYUV(img)
        # extract the lesion
        lesion = Caracteristics.extractLesion(img, contour)
        # convert img to gray
        lesion = cv2.cvtColor(lesion,cv2.COLOR_RGB2GRAY)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # crop the rect
        lesion = lesion[y:y + h, x:x + w]
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # sum of pixels
        s = np.sum(lesion)
        # get mean color value
        mean = s // lesionArea
        # calculate SDG
        lesion[lesion != 0] = np.subtract(lesion[lesion != 0], mean)
        lesion = np.power(lesion, 2)
        SDG = np.sum(lesion)
        # SDG = 0
        # for i in range(0, h):
        #     for j in range(0, w):
        #         if lesion[i, j] != 0:
        #             SDG = SDG + ((lesion[i, j] - mean)**2)
        SDG = np.sqrt((1 / lesionArea) * SDG)
        SDG = round(SDG, 2)
        return SDG

    @staticmethod
    def colorKurtosis(img, contour):
        '''
            Kurtosis, color distribution
        '''
        lesion = Caracteristics.extractLesion(img, contour)
        lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2GRAY)
        area = cv2.contourArea(contour)
        summ = np.sum(lesion)
        mean = summ/area
        lesion[lesion != 0] = np.subtract(lesion[lesion != 0], mean)
        lesion = np.power(lesion, 4)
        kurtosis = np.sum(lesion) / area
        kurtosis = round(kurtosis, 2)
        return kurtosis
