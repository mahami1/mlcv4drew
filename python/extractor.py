# -----------------------------------------------------------------------------
# Description: ROIExtractor implementations
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports

import cv2
import numpy as np
import argparse

from datetime import datetime
from interface import ROIExtractor
from data import Scene, Rectangle
from utility import LayoutUtility
from scenes import SceneReader, SceneGenerator
from observer import PerformanceObserver, ImageDecoratorObserver, CompositeObserver 

# -----------------------------------------------------------------------------
# Definitions

class HoughTransformROIExtractor(ROIExtractor):

    def __init__(self):
        self.image = None
        # Hough transform parameters
        # https://docs.opencv.org/3.4/d3/de5/tutorial_js_houghcircles.html
        self.minDist = 100
        self.minRadius = 5 
        self.maxRadius = 30
        self.param1 = 100
        self.param2 = 50
        self.ksize = 5

    def extractROIs(self):
        # Convert to gray-scale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        blurred = cv2.medianBlur(gray, self.ksize)
        # Apply hough transform on the image
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, 
                                   minDist=self.minDist, 
                                   param1=self.param1, param2=self.param2, 
                                   minRadius=self.minRadius, maxRadius=self.maxRadius)
        # The vaues form the Example
        # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)        
        rois = []
        if circles is not None:
            for circle in circles[0, :]:
                rois.append(self._getBBox(circle))

        return rois 

    def _getBBox(self, circle):
        x, y, r = circle
        top = np.round(y - r)
        left = np.round(x - r)
        width = height = np.round(2 * r)
        return Rectangle(left, top, width, height)

    def startAssembly(self, scene):
        if isinstance(scene, Scene):
            self.image = scene.image
        else:
            self.image = scene
   
    def finishAssembly(self, layout):
        self.image = None

# Evaluate performace of the Extrator over specified scenes 
# Returns: ModelStats
def evaluatePerformance(sceneDirs, outDir = None, silent = False, extractor = HoughTransformROIExtractor()):
    sceneSource = SceneReader(sceneDirs)
    observer = PerformanceObserver(silent = silent)
    if silent or outDir == None:
        decorator = None
    else:    
        decorator = ImageDecoratorObserver(outDir)
        # Leave only Labels and initial ROIs
        decorator.initialLayoutColor = None
        decorator.proposedROIcolor = None
        decorator.layoutExtensionColor = None
        decorator.finalLayoutColor = None

    while sceneSource.hasNext():
        scene = sceneSource.nextScene()

        extractor.startAssembly(scene)
        observer.startAssembly(scene)
        if decorator != None:
            decorator.startAssembly(scene)

        ROIs = extractor.extractROIs()
        observer.setInitialROIs(ROIs)
        if decorator != None:
            decorator.setInitialROIs(ROIs)

        observer.setInitialLayout(LayoutUtility.getScrews(ROIs))
        if decorator != None:
            decorator.finishAssembly(scene.labels)

    return observer.baseModelStats

# Optimise the Hough Transform parameters for the given scenes 
# in order to maximise the f1score
def optimiseF1socore(scenes):
    bestStats = None
    bestExtractor = None
    print(str(datetime.now()))
    for minDist in range(5, 105, 20):
        for minRadius in range(20, 50, 10):
            for maxRadius in range(minRadius + 10, 100, 20):
                for param1 in range(20, 100, 20):
                    # Running reporting
                    if bestStats != None and bestExtractor != None:
                        reportPerformance(bestStats, bestExtractor, final = False)                    
                    for param2 in range(20, 100, 20):
                        for ksize in [3, 5, 7]:
                            print(".", end="", flush=True)
                            extractor = HoughTransformROIExtractor()
                            extractor.minDist = minDist
                            extractor.minRadius = minRadius
                            extractor.maxRadius = maxRadius
                            extractor.param1 = param1
                            extractor.param2 = param2
                            extractor.ksize = ksize
                            stats = evaluatePerformance(scenes, silent=True, extractor=extractor)
                            if bestStats == None or stats.getMeanF1score() > bestStats.getMeanF1score():
                                bestStats = stats
                                bestExtractor = extractor
    reportPerformance(bestStats, bestExtractor, final = True)
    return bestStats, bestExtractor

def reportPerformance(bestStats, bestExtractor, final = False):
    print("")
    print(str(datetime.now()))
    if final == True:
        print("Final report:")
    else:
        print("Interim report:")

    print("Best f1score = {:.2f}".format(round(bestStats.getMeanF1score(), 2)) 
          + ", precision = {:.2f}".format(round(bestStats.getMeanPrecision(), 2))
          + ", recall = {:.2f}".format(round(bestStats.getMeanRecall(), 2)))
    print("  minDist = " + str(bestExtractor.minDist))
    print("  minRadius = " + str(bestExtractor.minRadius))
    print("  maxRadius = " + str(bestExtractor.maxRadius))
    print("  param1 = " + str(bestExtractor.param1))
    print("  param2 = " + str(bestExtractor.param2))
    print("  ksize = " + str(bestExtractor.ksize))

# -----------------------------------------------------------------------------
# Main - Test the extractor performance

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="extractor.py", description="Tests the performance of the HoughTransformROIExtractor "
                                     "over specified set of labelled disassembly scenes")

    parser.add_argument("-scenes", required=True, nargs='+', # type = readable directory, 
                        help="Input scene directories. Images and labels must be in sub-directory 'Images' and 'Annotations' repectively")

    parser.add_argument("-output", required=False, # type = writable directory, 
                        help="Directory to store the labeled scene images showing the detected circles and the labeled screws")

    parser.add_argument("-optimise", required=False, action="store_true", default=False, help="Optimises the Hough extractor parameters for maximum perfomance (very slow)")

    args = parser.parse_args()

    if args.optimise:
        optimiseF1socore(args.scenes)
    else:
        evaluatePerformance(args.scenes, args.output)

    print("Done")
