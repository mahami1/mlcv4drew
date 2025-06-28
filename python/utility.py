# -----------------------------------------------------------------------------
# Description: Various utility classes and functions.
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports

import cv2
import numpy as np

from data import Point, Screw, Transform2D

# -----------------------------------------------------------------------------
# Definitions

class LayoutUtility:

    def match(actualLayout, detectedLayout, maxOffset = None):
        if maxOffset == None:
            meanDiameter = LayoutUtility.getMeanDiameter(actualLayout)
            meanDiameter += LayoutUtility.getMeanDiameter(detectedLayout)
            maxOffset = int(np.round(meanDiameter / 2.0))

        result = []
        detectedLayout = detectedLayout.copy()
        for actual in actualLayout:
            for detected in detectedLayout:
                if actual.getDistance(detected) < maxOffset :
                    result.append([actual, detected])
                    detectedLayout.remove(detected) 
                    break

        return result

    def transform(layout, transform): 
        result = []
        for screw in layout:
            newCenter = transform.apply(screw.center)
            result.append(Screw(newCenter, screw.diameter))
        return result; 

    def getMeanDiameter(layout):
        sum = 0.0
        for screw in layout:
            sum += screw.diameter
        return int(np.round(sum / len(layout))) if len(layout) > 0 else 0
    
    def getROIs(screws):
        result = []
        for screw in screws:
            result.append(screw.getBBox())
        return result

    def getScrews(rois):
        result = []
        for bbox in rois:
            result.append(Screw.fromBBox(bbox))
        return result

class PatternUtility:
    
    # Description: Attempts to move the proposed layout over the actual layout in order to find the 
    #   best possible match. This is done by:
    #
    #   1. finding intervals in the proposed layout that are presend in the actual/detected layout
    #   2. finding affine transformatiion (translation + rotation) that moves each of the the matching 
    #      intervals from the proposed layout to the corrsponding actual layout interval
    #   3. apply the same transformation to the whole prposed layout and finding how many screws will 
    #      match (be at distance < tolerance).
    #   4. Score each match by calculating number of matched screws / total number of screws in the 
    #      actual layout
    #   5. Return the transformed proposed laysout that have maximum matched screws with the actual 
    #      layout
    #
    # Arguments:
    #   proposedLayout - ScrewLayout instance of the proposed layout (from the learned layouts repository self.layouts)
    #   actualLayout - ScrewLayout instance of the detected layout
    # Returns: (match, score)
    #   Screw[] match - list of screws representing the transformed proposed layout that gives best match
    #   double score - the score of the match (0 to 1 (best))
    #   Transform2D transform - (optional) the afinne transform used to produce the best match
    #
    def getBestMatch(proposedLayout, actualLayout):

        bestMatch = [] 
        bestScore = 0.0
        bestTransform = None 
        
        tolerance = LayoutUtility.getMeanDiameter(actualLayout.screws + proposedLayout.screws);  
        
        for distance in actualLayout.getUniqueDistances(tolerance):
            for proposalPair in proposedLayout.getMatchingIntervals(distance, tolerance, True):
                for actualPair in actualLayout.getMatchingIntervals(distance, tolerance):
                    transform = PatternUtility.getTransform(proposalPair, actualPair)
                    # Debug: the transform shold transform target source to target
                    at = transform.apply(proposalPair.points[0])
                    bt = transform.apply(proposalPair.points[1])
                    # ---
                    match = LayoutUtility.transform(proposedLayout.screws, transform)
                    # matchLayout = ScrewLayout(match) # Debug
                    score = PatternUtility.getMatchScore(match, actualLayout.screws, tolerance)
                    if score > bestScore:
                        bestMatch = match
                        bestScore = score
                        bestTransform = transform

        return bestMatch, bestScore, bestTransform

    def getTransform(fromInterval, toInterval):
        transform = Transform2D()
        
        # Get the middle of each interval
        fromMiddle = fromInterval.getMiddle()
        toMiddle = toInterval.getMiddle()

        # Let the toInterval middle be the anchor point
        anchor = toMiddle
        
        # first, translate the fromInterval middle point to the anchor
        dx = anchor.x - fromMiddle.x
        dy = anchor.y - fromMiddle.y
        transform = transform.translate(dx, dy)

        # then, rotate fromInterval around the anchor so it alligns with toInterval 
        transform = transform.rotateRadiansAroundPoin(anchor, toInterval.getIncline() - fromInterval.getIncline())
        # test = result.apply(fromInterval.points[0]) # debug
        return transform

    def getMatchScore(proposedLayout, actualLayout, maxOffset = None):
        matchedPairs = LayoutUtility.match(proposedLayout, actualLayout, maxOffset)
        return len(matchedPairs) / float(len(actualLayout))
