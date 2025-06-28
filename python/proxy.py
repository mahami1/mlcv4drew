# -----------------------------------------------------------------------------
# Description: TODO
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports

import random
import numpy as np
import argparse

from main import TwoPassScrewDetector
from observer import PerformanceObserver
from proposer import PatternMatchingProposer, ListPatternStore, LearningPatternStore
from utility import LayoutUtility
from data import Rectangle, Scene, Screw, Point, Transform2D
from interface import ROIExtractor, ScrewDetector, SceneSource

# -----------------------------------------------------------------------------
# Definitions

class ProxySceneSource(SceneSource):

    def __init__(self, count, rotate=True, minOffset=-10, maxOffset=10):
        self.count = count
        self.rotate = rotate
        self.minOffset = minOffset
        self.maxOffset = maxOffset

    def hasNext(self):
        return self.count > 0

    def nextScene(self):
        if not self.hasNext():
            return None
        
        choice = random.randint(0, 2)
        result = None
        if choice == 0:
            result = self._getTriangle()
        elif choice == 1:
            result = self._getRectangle()
        else:
            result = self._getHexagon()

        transform = Transform2D()
        if self.rotate:
            transform = transform.rotateDegree(random.randint(-180, 180))

        if self.minOffset > 0 or self.maxOffset > 0:
            dx = random.randint(self.minOffset, self.maxOffset)
            dy = random.randint(self.minOffset, self.maxOffset)
            transform = transform.translate(dx, dy)

        result = LayoutUtility.transform(result, transform)
        self.count -= 1

        return Scene(None, result)

    def getAll(self):
        result = []
        result.append(self._getTriangle())
        result.append(self._getRectangle())
        result.append(self._getHexagon())
        return result

    def _getTriangle(self):
        result = []
        result.append(Screw(Point(100, 100), 6))
        result.append(Screw(Point(200, 100), 6))
        result.append(Screw(Point(100, 200), 6))
        return result

    def _getRectangle(self):
        result = []
        result.append(Screw(Point(50, 50), 6))
        result.append(Screw(Point(100, 50), 6))
        result.append(Screw(Point(50, 100), 6))
        result.append(Screw(Point(100, 100), 6))
        return result

    def _getHexagon(self):
        result = []
        result.append(Screw(Point(0, 60), 6))
        result.append(Screw(Point(-60, 0), 6))
        result.append(Screw(Point(60, 0), 6))
        result.append(Screw(Point(-60, -60), 6))
        result.append(Screw(Point(60, -60), 6))
        result.append(Screw(Point(0, 120), 6))
        return result

class ProxyROIExtractor(ROIExtractor):

    class FalsePositiveGenerateor:

        def __init__(self, sceneBBox, minDiameter, maxDiameter):
            self.sceneBBox = sceneBBox
            self.minDiameter = minDiameter
            self.maxDiameter = maxDiameter

        def getRandomROI(self):
            dx = random.randint(0, int(self.sceneBBox.width) - self.maxDiameter)
            dy = random.randint(0, int(self.sceneBBox.height) - self.maxDiameter)
            diameter = random.randint(self.minDiameter, self.maxDiameter); 
            
            left = int(np.round(self.sceneBBox.left + dx))
            top = int(np.round(self.sceneBBox.top + dy))
            
            result = Rectangle(left, top, diameter, diameter);
            return result; 

    def __init__(self, recall, precision = None, meanOffset = 2):
        self.recall = recall
        if precision == None:
            precision = recall
        self.precision = precision
        self.meanOffset = meanOffset

    def startAssembly(self, scene):
        self.actualLayout = []
        for screw in scene.labels:
            self.actualLayout.append(screw.getBBox())
        dmin, dmax = self._getDiameterRange()            
        self.generator = ProxyROIExtractor.FalsePositiveGenerateor(
            self._getSceneBBox(), dmin, dmax)
   
    def finishAssembly(self, layout):
        pass

    def extractROIs(self):
        result = []
        for roi in self.actualLayout:
            if random.randint(0, 100) < (100 * self.recall):
                # true positives
                detected = self._getRandomOffset(roi)
                result.append(detected)
            else:
                # false negative
                pass
		
		# Generate some random noise (false positives)
        falsePositiveRate = 1 - self.precision
        if falsePositiveRate > 0:
            falsePositiveCount = random.randint(0, int(np.round(len(result) * falsePositiveRate)))
            for i in range(falsePositiveCount):
                result.append(self.generator.getRandomROI());
        
        return result
    
    def _getRandomOffset(self, roi):
        if self.meanOffset > 0:
            dx = random.randint(-self.meanOffset, self.meanOffset)
            dy = random.randint(-self.meanOffset, self.meanOffset)
            return Rectangle(roi.left + dx, roi.top + dy, roi.width, roi.height)
        return roi

    def _getDiameterRange(self):
        min = None
        max = 0
        for bbox in self.actualLayout:
            d = np.min([bbox.width, bbox.height])
            if min == None or min > d:
                min = d
            if max < d:
                max = d    
        return min, max

    def _getSceneBBox(self):
        left = top = bottom = right = None
        for bbox in self.actualLayout:
            bbox.bottom = bbox.top + bbox.height
            bbox.right = bbox.left + bbox.width
            if left == None or left > bbox.left:
                left = bbox.left
            if top == None or top > bbox.top:
                top = bbox.top
            if bottom == None or bottom < bbox.bottom:
                bottom = bbox.bottom
            if right == None or right < bbox.right:
                right = bbox.right
        return Rectangle(left, top, right - left, bottom - top)

class ProxyScrewDetector(ScrewDetector):

    def __init__(self, recall, precision = None):
        self.recall = recall
        if precision == None:
            precision = recall
        self.precision = precision

    def startAssembly(self, scene):
        self.actualLayout = scene.labels
   
    def finishAssembly(self, layout):
        pass

    def detectScrews(self, rois): 
        result = []
        falsePositives = LayoutUtility.getScrews(rois)
        falseNegatives = self.actualLayout.copy()
        
        truePositives = self._match(falseNegatives, falsePositives); 
        
        for screw in truePositives:
            if random.randint(0, 100) < (100 * self.recall):
                # true positive
                result.append(screw)
            else:
                # false negative
                pass
        
        for screw in falsePositives:
            if random.randint(0, 100) > (100 * self.precision):
                # add some false positives
                result.append(screw); 
        
        return result

    def _match(self, actualLayout, detectedLayout):
        matched = []
        for actual, detected in LayoutUtility.match(actualLayout, detectedLayout):
            actualLayout.remove(actual)
            detectedLayout.remove(detected)
            matched.append(detected)
        return matched

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="proxy.py", description="Tests the performance of the two-pass screw detection algorithm "
                                     "by using configurable quality proxy implementations of the ROI-extractor and the screw-detector")

    parser.add_argument("-count", type=int, required=True, help="number of simulated scenes")
    parser.add_argument("-ep", type=float, required=True, help="ROI extractor precision value (range 0..1)")
    parser.add_argument("-er", type=float, required=False, help="ROI extractor recall value (range 0..1). If ot provided, recall will be same as the precision.")
    parser.add_argument("-dp", type=float, required=True, help="Screw detector precision value (range 0..1)")
    parser.add_argument("-dr", type=float, required=False, help="Screw detector recall value (range 0..1). If ot provided, recall will be same as the precision.")
    parser.add_argument("-learn", required=False, action="store_true", default=False, help="Use lerning pattern store")
    parser.add_argument("-seed", type=int, required=False, help="The random generator seed value")

    args = parser.parse_args()

    if args.er == None:
        args.er = args.ep

    if args.dr == None:
        args.dr = args.dp

    if args.seed != None:
        random.seed(args.seed)

    sceneSource = ProxySceneSource(count = args.count)
    extractor = ProxyROIExtractor(recall = args.er, precision = args.ep) 
    detector = ProxyScrewDetector(recall = args.dr, precision = args.dp) 
    
    if args.learn:
        store = LearningPatternStore()
    else:        
        # preload the layouts
        store = ListPatternStore()
        for layout in sceneSource.getAll():
            store.addLayout(layout)
        store.locked = True

    proposer = PatternMatchingProposer(store)
    observer = PerformanceObserver() 

    worker = TwoPassScrewDetector(extractor, proposer, detector, observer)

    while sceneSource.hasNext():
        layout = worker.extractScrewLayout(sceneSource.nextScene())
        print("")
    
    print("Done")
