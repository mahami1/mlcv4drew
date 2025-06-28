# -----------------------------------------------------------------------------
# Description: Observer implementation that evaluates the model prformance
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports

import cv2
import numpy as np

from data import Screw, Scene, Point, Rectangle
from interface import Observer
from utility import LayoutUtility

# -----------------------------------------------------------------------------
# Definitions

class Statistics:
    def __init__(self):
        pass

class ModelPerformance:
    def __init__(self):
        self.clear()

    def clear(self):
        self.precision = []
        self.recall = []
        self.f1score = []
        self.offset = []

    def getMeanPrecision(self):
        return np.mean(self.precision)

    def getMeanRecall(self):
        return np.mean(self.recall)
    
    def getMeanOffset(self):
        return np.mean(self.offset)
    
    def getMeanF1score(self):
        return np.mean(self.f1score)

class ScrewMatch:
    def __init__(self, actualScrew, detectedScrew):
        self.actualScrew = actualScrew
        self.detectedScrew = detectedScrew
        
    def getOffset(self):
        return int(self.actualScrew.getCenter().distance(self.detectedScrew.getCenter()))

class PerformanceObserver(Observer):

    def __init__(self, maxOffset = 5, silent = False):
        self.maxOffset = maxOffset
        self.silent = silent
        self.baseModelStats = ModelPerformance()
        self.modelStats = ModelPerformance()
    
    def startAssembly(self, scene):
        self.actualLayout = scene.labels
        if self.actualLayout != None:
            self.maxOffset = LayoutUtility.getMeanDiameter(self.actualLayout)
   
    def setInitialROIs(self, rois):
        pass
    
    def setInitialLayout(self, layout):
        self._calculatePerformance(layout, self.baseModelStats)
        if self.silent == False:
            self._reportPerformance(self.baseModelStats, "BaseM")
    
    def setProposedROIs(self, rois):
        pass
    
    def setLayoutExtension(self, layout):
        pass

    def finishAssembly(self, layout):
        self._calculatePerformance(layout, self.modelStats)
        if self.silent == False:
            self._reportPerformance(self.modelStats, "Model")

    # Description: Updated the model statistics
    # Arguments:
    #   detectedLayout - list of Screw instances
    #   modelStats - the perfomance statistics instance for the model
    # Returns: none
    #
    def _calculatePerformance(self, detectedLayout, modelStats):
        truePositives, falseNegatives, falsePositives = self._match(self.actualLayout, detectedLayout)
		
        precision = len(truePositives) / (len(truePositives) + len(falsePositives)) if len(truePositives) > 0 else 0
        recall = len(truePositives) / (len(truePositives) + len(falseNegatives)) if len(truePositives) > 0 else 0
        f1score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        offsets = []
        for match in truePositives:
            offset = match.actualScrew.getDistance(match.detectedScrew)
            offsets.append(offset)
		
        modelStats.precision.append(precision)
        modelStats.recall.append(recall)
        modelStats.f1score.append(f1score)
        modelStats.offset.append(np.mean(offsets) if len(offsets) > 0 else 0)

    def _reportPerformance(self, modelStats, modelLabel):
        print(modelLabel + ": " 
              + "precision={:.2f}".format(round(modelStats.precision[-1], 2)) + "({:.2f}), ".format(round(modelStats.getMeanPrecision(), 2))
              + "recall={:.2f}".format(round(modelStats.recall[-1], 2)) + "({:.2f}), ".format(round(modelStats.getMeanRecall(), 2))
              + "f1score={:.2f}".format(round(modelStats.f1score[-1], 2)) + "({:.2f}), ".format(round(modelStats.getMeanF1score(), 2)) 
              + "offset={:.2f}".format(round(modelStats.offset[-1], 2)) + "({:.2f})".format(round(modelStats.getMeanOffset(), 2))) 
    

    # Description: Separates the screws into matched, false-positives and false-negatives
    # Arguments:
    #   actual - contais list of actual screws, after return will contain the 
    #   detected - contains the list of detected screws, after return will contain false-positives 
    # Returns: (matches, falseNegatives, falsePositives)
    #   match - list of screw matches
    #   falseNegatives - list of false-negattives
    #   falsePositives - list of false positives
    #
    def _match(self, actualScrews, detectedScrews):
        matches = []
        falseNegatives = actualScrews.copy()
        falsePositives = detectedScrews.copy()

        for actual in actualScrews :
            for detected in detectedScrews:
                if actual.center.getDistance(detected.center) < self.maxOffset :
                    matches.append(ScrewMatch(actual, detected))
                    if actual in falseNegatives : falseNegatives.remove(actual)
                    if detected in falsePositives : falsePositives.remove(detected)

        return matches, falseNegatives, falsePositives

class ImageDecoratorObserver(Observer):

    # green: (0, 255, 0)    - Manual Label
    # red: (0, 0, 255)      - Proposed ROI
    # blue: (255, 0, 0)     - Inital Layout
    # white: (255, 255, 255) 
    # yellow: (0, 255, 255) - Initial ROI
    # cyan: (255, 255,0)    - Layout extension
    # magenta: (255, 0, 255) - Final layout

    def __init__(self, outDir):
        self.outDir = outDir
        self.sceneId = 0
        self.image = None
        # Reset counters
        self.labelCount = 0
        self.initialLayoutCount = 0
        self.initialROIcount = 0
        self.proposedROIcount = 0
        self.layoutExtensionCount = 0
        self.finalLayoutCount = 0
        # Set attribute color to None to prevent displayng it
        self.labelColor = (0,255,0) # green
        self.initialROIcolor = (0, 255, 255) # yellow
        self.initialLayoutColor = (255, 0, 0) # blue
        self.proposedROIcolor = (0, 0, 255) # red
        self.layoutExtensionColor = (255, 255,0) # Cyan
        self.finalLayoutColor = (255, 0, 255) # magenta 

    def startAssembly(self, scene):
        if isinstance(scene, Scene):
            self.image = scene.image.copy()
            self.labelCount = len(scene.labels)
            for screw in scene.labels:
                self._drawBox(screw.getBBox(), self.labelColor)
        else:
            self.image = scene.copy()
            self.labelCount = 0

    def setInitialROIs(self, rois):
        self.initialROIcount = len(rois)
        for roi in rois:
            self._drawBox(roi, self.initialROIcolor, offset=-4)
    
    def setInitialLayout(self, layout):
        self.initialLayoutCount = len(layout)
        for screw in layout:
            self._drawScrew(screw, self.initialLayoutColor, offset=-4)
    
    def setProposedROIs(self, rois):
        self.proposedROIcount = len(rois)
        for roi in rois:
            self._drawBox(roi, self.proposedROIcolor, offset=4)

    def setLayoutExtension(self, layout):
        self.layoutExtensionCount = len(layout)
        for screw in layout:
            self._drawScrew(screw, self.layoutExtensionColor, offset=4)

    def finishAssembly(self, layout):
        self.finalLayoutCount = len(layout)
        for screw in layout:
            self._drawScrew(screw, self.finalLayoutColor)
        self._drawLegend()
        file = self.outDir + "/" + str(self.sceneId).zfill(3) + ".png"
        cv2.imwrite(file, self.image)
        self.image = None
        self.sceneId += 1

    def _drawBox(self, bbox, color, tickness = 2, offset = 0):
        if color == None:
            return
        p1 = (int(bbox.left + offset), int(bbox.top + offset))
        p2 = (int(bbox.left + bbox.width + offset), int(bbox.top + bbox.height + offset))
        self.image = cv2.rectangle(self.image, p1, p2, color, tickness)

    def _drawScrew(self, screw, color, tickness = 2, offset = 0):
        if color == None:
            return
        radius = int(screw.diameter // 2)
        center = (int(screw.center.x + offset), int(screw.center.y + offset))
        self.image = cv2.circle(self.image, center, radius, color, tickness)

    def _drawLegend(self):
        legendWidth = 600
        legendHeight = 300
        imageWidth = self.image.shape[1]
        imageHeight = self.image.shape[0]
        self.legendColor = (255, 255, 255) # while
        legendBBox = Rectangle(int(imageWidth - legendWidth), int(imageHeight - legendHeight), legendWidth, legendHeight)
        self._drawBox(legendBBox, self.legendColor)
        offset = Point(legendBBox.left + 10, legendBBox.top + 10)

        offset = self._drawLegendItem("Labels (" + str(self.labelCount) + ")", self.labelColor, offset, box=True)
        offset = self._drawLegendItem("Initial ROI (" + str(self.initialROIcount) + ")", self.initialROIcolor, offset, box=True)
        offset = self._drawLegendItem("Initial layout (" + str(self.initialLayoutCount) + ")", self.initialLayoutColor, offset)
        offset = self._drawLegendItem("Proposed ROI (" + str(self.proposedROIcount) + ")", self.proposedROIcolor, offset, box=True)
        offset = self._drawLegendItem("Proposed layout extension (" + str(self.layoutExtensionCount) + ")", self.layoutExtensionColor, offset)
        offset = self._drawLegendItem("Final layout (" + str(self.finalLayoutCount) + ")", self.finalLayoutColor, offset)
    
    def _drawLegendItem(self, text, color, origin, box = False):

        if color == None:
            return origin
    
        if box == True:
            self._drawBox(Rectangle(origin.x, origin.y, 30, 30), color)
        else: 
            self._drawScrew(Screw(Point(origin.x + 15, origin.y + 17), 35), color)

        self.image = cv2.putText(self.image, text, org = (origin.x + 40, origin.y + 30), 
                    fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0, 
                    color = color, thickness = 2)        

        return Point(origin.x, origin.y + 40)

class CompositeObserver(Observer):

    def __init__(self, observers):
        self.observers = observers

    def startAssembly(self, scene):
        for observer in self.observers:
            observer.startAssembly(scene)

    def setInitialROIs(self, rois):
        for observer in self.observers:
            observer.setInitialROIs(rois)
    
    def setInitialLayout(self, layout):
        for observer in self.observers:
            observer.setInitialLayout(layout)
    
    def setProposedROIs(self, rois):
        for observer in self.observers:
            observer.setProposedROIs(rois)
    
    def setLayoutExtension(self, layout):
        for observer in self.observers:
            observer.setLayoutExtension(layout)

    def finishAssembly(self, layout):
        for observer in self.observers:
            observer.finishAssembly(layout)
