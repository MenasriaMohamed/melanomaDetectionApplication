import cv2
from .utils.Contours import Contours
from .utils.Asymmetry import Asymmetry
from .utils.Border import Border
from .utils.Color import Color
from .utils.Diameter import Diameter
from .utils.SevenPointsChecklist import SevenPointsChecklist
from .utils.Menzies import Menzies

class Caracteristics:
    @staticmethod
    def extractCaracteristics(imgPath):
        '''
            extracts and returns the caracteristics of an img
        '''
        # read the img
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        contour = Contours.contours2(img)
        car = {}
        # append asymmetrys
        car['car0'] = Asymmetry.asymmetryByBestFitEllipse(img, contour)
        car['car1'] = Asymmetry.asymmetryByDistanceByCircle(img, contour)
        car['car2'] = Asymmetry.asymmetryIndex(img, contour)
        car['car3'] = Asymmetry.asymmetryBySubRegion(img, contour)
        car['car4'] = Asymmetry.asymmetryBySubRegionCentered(img, contour)
        car['car5'] = Asymmetry.asymmetryBySubRegionCentered2(img, contour)
        # append borders
        car['car6'] = Border.borderRoundness(img, contour)
        car['car7'] = Border.borderLength(img, contour)
        car['car8'] = Border.borderRegularityIndex(contour)
        car['car9'] = Border.borderRegularityIndexRatio(img, contour)
        car['car10'] = Border.borderCompactIndex(contour)
        car['car11'] = Border.borderHeywoodCircularityIndex(img, contour)
        car['car12'] = Border.borderHarrisCorner(img, contour)
        car['car13'] = Border.borderFractalDimension(img, contour)
        # append colros
        car['car14'] = Color.colorHSVIntervals(img, contour)
        car['car15'] = Color.colorYUVIntervals(img, contour)
        car['car16'] = Color.colorYCbCrIntervals(img, contour)
        car['car17'] = Color.colorSDG(img, contour)
        car['car18'] = Color.colorKurtosis(img, contour)
        # append diameters
        car['car19'] = Diameter.diameterMinEnclosingCircle(img, contour)
        car['car20'] = Diameter.diameterOpenCircle(img, contour)
        car['car21'] = Diameter.diameterLengtheningIndex(img, contour)
        # # 7 points
        car['car22'] = SevenPointsChecklist.inflammationAndBloodness(img, contour)
        car['car23'] = SevenPointsChecklist.sensibility(img, contour)
        # # menzies
        car['car24'] = Menzies.darkPoints(img, contour)
        car['car25'] = Menzies.blueGrey(img, contour)
        return car
