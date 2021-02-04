import cv2
import numpy as np
import imutils
from scipy import ndimage
from .Preprocess import Preprocess
from .Caracteristics import Caracteristics

class Asymmetry:
    '''
        all Asymmetry methods
    '''
    @staticmethod
    def asymmetryByBestFitEllipse(img, contour):
        '''
            get asymmetry by best fitted ellipse
        '''
        # remove artifact
        img = Preprocess.removeArtifactYUV(img)
        # convert img to gray
        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # get best fitted ellipse
        ellipse = cv2.fitEllipse(contour)
        # blank img
        blankImg = np.zeros(np.shape(imgray))
        # draw ellipse on blank img
        cv2.ellipse(blankImg, ellipse, (255, 255, 255), -1)
        # get ellipse area
        ellipseArea = np.sum(blankImg != 0)
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # diff between lesion and ellipse area
        delta = abs(ellipseArea - lesionArea)
        asymmetry = (delta / ellipseArea) * 100
        asymmetry = round(asymmetry, 2)
        return asymmetry

    @staticmethod
    def asymmetryByDistanceByCircle(img, contour):
        '''
            distance Between the center of gravity of contour and  center of circle around the contour
        '''
        # get moment of contour
        M = cv2.moments(contour)
        # get center of gravity of contour
        xe = int(M["m10"] / M["m00"])
        ye = int(M["m01"] / M["m00"])
        # get center of circle around the contour
        cv2.circle(img, (xe, ye), radius=2, color=(0, 255, 255), thickness=1)
        (xCiCe, yCiCe), radius = cv2.minEnclosingCircle(contour)
        xCiCe = int(xCiCe)
        yCiCe = int(yCiCe)
        cv2.circle(img, (xCiCe, yCiCe), radius=2, color=(0, 0, 255), thickness=1)
        asm = 100 - (Caracteristics.DistanceEuclidean(xe, ye, xCiCe, yCiCe) * 100 / radius)
        asm = round(asm, 2)
        return asm

    @staticmethod
    def asymmetryIndex(img, contour):
        '''
            get asymmetry index
            search for homologue of each pixel
        '''
        # remove artifact
        img = Preprocess.removeArtifactYUV(img)
        # convert img to gray
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # crop the rect
        rect = imgray[y:y + h, x:x + w]
        # rotate 180°
        rotated = imutils.rotate_bound(rect, 180)
        # intersection between rect and rotated (search)
        intersection = cv2.bitwise_and(rect, rotated)
        imgray[y:y + h, x:x + w] = intersection
        # get area of intersection (black means no homologues found)
        intersectionArea = np.sum(intersection != 0)
        noHomologueArea = np.sum(intersection == 0)
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # asymmetry
        asymmetry = (noHomologueArea / lesionArea) * 100
        asymmetry = round(asymmetry, 2)
        return asymmetry
    
    @staticmethod
    def asymmetryBySubRegion(img, contour):
        '''
            get asymmetry by dividing the lesion to 4 subregions
        '''
        # remove artifact
        img = Preprocess.removeArtifactYUV(img)
        # convert img to gray
        img = Caracteristics.extractLesion(img, contour)
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # binarize the img
        # imgray[imgray > 0] = 255
        # find best fit ellipse
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        padding = 0
        # crop the rect
        rect = imgray[y - padding:y + h + padding, x - padding:x + w + padding]
        # rotate the lesion according to its best fit ellipse
        rect = ndimage.rotate(rect, angle, reshape=True)
        # flip H, flip V, flip VH
        rectH = cv2.flip(rect, 0)
        rectV = cv2.flip(rect, 1)
        rectVH = cv2.flip(rect, -1)
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # intersect rect and rectH
        intersection1 = cv2.bitwise_and(rect, rectH)
        intersectionArea1 = np.sum(intersection1 != 0)
        result1 = (intersectionArea1 / lesionArea) * 100
        # intersect rect and rectV
        intersection2 = cv2.bitwise_and(rect, rectV)
        intersectionArea2 = np.sum(intersection2 != 0)
        result2 = (intersectionArea2 / lesionArea) * 100
        # intersect rect and rectVH
        intersection3 = cv2.bitwise_and(rect, rectVH)
        intersectionArea3 = np.sum(intersection3 != 0)
        result3 = (intersectionArea3 / lesionArea) * 100
        res = [result1, result2, result3]
        asymmetry = max(res)
        asymmetry = 100 - asymmetry
        asymmetry = round(asymmetry, 2)
        return asymmetry
    
    @staticmethod
    def asymmetryBySubRegionCentered(img, contour):
        '''
            get asymmetry by dividing the lesion to 4 subregions
            but the lesion is placed in the center of img
        '''
        # remove artifact
        img = Preprocess.removeArtifactYUV(img)
        # convert img to gray
        img = Caracteristics.extractLesion(img, contour)
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # binarize the img
        # imgray[imgray > 0] = 255
        # find best fit ellipse
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # get moments of contour
        M = cv2.moments(contour)
        # center of gravity of the lesion
        xe = int(M["m10"] / M["m00"])
        ye = int(M["m01"] / M["m00"])
        # get the centered rect
        cx = x + w//2
        cy = y + h//2
        deltaX1 = abs(int(xe - cx))
        deltaY1 = abs(int(ye - cy))
        x1 = x + deltaX1
        w1 = w + deltaX1
        y1 = y + deltaY1
        h1 = h + deltaY1
        padding = 0
        # crop the rect
        rect = imgray[y1 - padding:y1 + h1 + padding, x1 - padding:x1 + w1 + padding]
        # rotate the lesion according to its best fit ellipse
        rect = ndimage.rotate(rect, angle, reshape=False)
        # flip H, flip V, flip VH
        rectH = cv2.flip(rect, 0)
        rectV = cv2.flip(rect, 1)
        rectVH = cv2.flip(rect, -1)
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # intersect rect and rectH
        intersection1 = cv2.bitwise_and(rect, rectH)
        intersectionArea1 = np.sum(intersection1 != 0)
        result1 = (intersectionArea1 / lesionArea) * 100
        # intersect rect and rectV
        intersection2 = cv2.bitwise_and(rect, rectV)
        intersectionArea2 = np.sum(intersection2 != 0)
        result2 = (intersectionArea2 / lesionArea) * 100
        # intersect rect and rectVH
        intersection3 = cv2.bitwise_and(rect, rectVH)
        intersectionArea3 = np.sum(intersection3 != 0)
        result3 = (intersectionArea3 / lesionArea) * 100
        res = [result1, result2, result3]
        asymmetry = max(res)
        asymmetry = 100 - asymmetry
        asymmetry = round(asymmetry, 2)
        return asymmetry
    
    @staticmethod
    def asymmetryBySubRegionCentered2(img, contour):
        '''
            get asymmetry by dividing the lesion to 4 subregions
            but the lesion is placed in the center of img
        '''
        # remove artifact
        img = Preprocess.removeArtifactYUV(img)
        # convert img to gray
        img = Caracteristics.extractLesion(img, contour)
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # binarize the img
        # imgray[imgray > 0] = 255
        # find best fit ellipse
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
        # get bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        # get moments of contour
        M = cv2.moments(contour)
        # center of gravity of the lesion
        xe = int(M["m10"] / M["m00"])
        ye = int(M["m01"] / M["m00"])
        # get the centered rect
        deltaX1 = abs(int(xe - x))
        deltaX2 = abs(int(xe - (x + w)))
        deltaY1 = abs(int(ye - y))
        deltaY2 = abs(int(ye - (y + h)))
        if deltaX1 < deltaX2:
            x1 = int(xe - deltaX2)
            w1 = int(deltaX2 * 2)
        else:
            x1 = int(xe - deltaX1)
            w1 = int(deltaX1 * 2)
        if deltaY1 < deltaY2:
            y1 = int(ye - deltaY2)
            h1 = int(deltaY2 * 2)
        else:
            y1 = int(ye - deltaY1)
            h1 = int(deltaY1 * 2)
        padding = 0
        # crop the rect
        rect = imgray[y1 - padding:y1 + h1 + padding, x1 - padding:x1 + w1 + padding]
        # rotate the lesion according to its best fit ellipse
        rect = ndimage.rotate(rect, angle, reshape=True)
        # flip H, flip V, flip VH
        rectH = cv2.flip(rect, 0)
        rectV = cv2.flip(rect, 1)
        rectVH = cv2.flip(rect, -1)
        # lesion area
        lesionArea = cv2.contourArea(contour)
        # intersect rect and rectH
        intersection1 = cv2.bitwise_and(rect, rectH)
        intersectionArea1 = np.sum(intersection1 != 0)
        result1 = (intersectionArea1 / lesionArea) * 100
        # intersect rect and rectV
        intersection2 = cv2.bitwise_and(rect, rectV)
        intersectionArea2 = np.sum(intersection2 != 0)
        result2 = (intersectionArea2 / lesionArea) * 100
        # intersect rect and rectVH
        intersection3 = cv2.bitwise_and(rect, rectVH)
        intersectionArea3 = np.sum(intersection3 != 0)
        result3 = (intersectionArea3 / lesionArea) * 100
        res = [result1, result2, result3]
        asymmetry = max(res)
        asymmetry = 100 - asymmetry
        asymmetry = round(asymmetry, 2)
        return asymmetry