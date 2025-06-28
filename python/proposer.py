# -----------------------------------------------------------------------------
# Description: ROIProposer implementations
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports
from abc import ABC, abstractmethod

from data import Point, Screw, Pattern
from interface import ROIProposer
from utility import LayoutUtility, PatternUtility

# -----------------------------------------------------------------------------
# Definitions

class PatternStore(ABC):
    #
    # Description: TODO.
    # Arguments:
    #   Pattern layout - the partial layout that will be matched against stored layout patterns 
    # Returns: (layout, store, transform = None)
    #   Screw[] layout - the layout that best fits/matches with the partial layout
    #   double  score  - ratio of the matched / all screws in the partial layout. 
    #   Transform2D trnsform - optional. The transform used to match the partial layout (input) to the returned full layout 
    #
    @abstractmethod
    def getBestMatch(self, layout):
        pass
    #
    # Description: Adds layout to the store. 
    # Arguments:
    #   Screw[] layout - the layout to be stored / memorised
    #
    def addLayout(self, layout):
        self.addPattern(Pattern(layout))

    #
    # Description: Adds pattern to the store. 
    # Arguments:
    #   Pattern pattern - the layout to be stored / memorised
    #
    @abstractmethod
    def addPattern(self, pattern):
        pass

class ListPatternStore(PatternStore):

    def __init__(self):
        self.patterns = []
        self.locked = False

    def getBestMatch(self, layout):
        bestMatch = []
        bestScore = 0.0
        bestTransform = None
        for proposal in self.patterns:
            match, score, transform = PatternUtility.getBestMatch(proposal, layout)
            if score > bestScore:
                bestMatch = match
                bestScore = score
                bestTransform = transform
        return bestMatch, bestScore, bestTransform

    def addPattern(self, pattern):
        if self.locked:
            return
        for knownPattern in self.patterns:
            match, score, _ = PatternUtility.getBestMatch(knownPattern, pattern)
            if score > 0.99:
                # perfect match - no need to add this pattern as we already have it in memory
                return
        # the pattern was not found into memory so let memorise it
        self.patterns.append(pattern)

class LearningPatternStore(PatternStore):

    def __init__(self, mergeThreshold = 0.9, forgetThreshold = 0.1, 
                 compactThreshold = 10, trailSize = 10, minPatternSize = 3):
        # if best match score is > than this, then pattern will be merged into the best match. 
        # Otherwise will be added as new pattern
        self.mergeThreshold = mergeThreshold 
        # if dynamic pattern matchRate is below this value, it will be removed from database:        
        self.forgetThreshold = forgetThreshold 
        # how often to compact database (every N number of pattern additions)
        self.compactThreshold = compactThreshold
        # when this reaches compactThreshold, compact and reset it
        self.compactCounter = 0
        # how many patterns to keep in the dynamic pattern history
        self.trailSize = trailSize
        # patterns with less than that screws will not be stored (no point decreasing that below 3)
        self.minPatternSize = minPatternSize
        # list of meorised patterns (DynamicPattern instances)
        self.patterns = []

    def getBestMatch(self, layout):
        bestMatch = None
        bestScore = 0
        bestTransform = None
        for pattern in self.patterns:
            match, score, transform = PatternUtility.getBestMatch(pattern, layout)
            if score > bestScore:
                bestMatch = match
                bestScore = score
                bestTransform = transform
        return bestMatch, bestScore, bestTransform

    def addPattern(self, pattern):
        if pattern.size() < self.minPatternSize:
            # ignore too small patterns
            return
        
        # Check if pattern is already memorised
        bestMatch = None 
        bestScore = 0
        targetPattern = None
        for memorised in self.patterns:
            match, score, _ = PatternUtility.getBestMatch(pattern, memorised)
            if score > bestScore:
                bestMatch = match
                bestScore = score
                targetPattern = memorised

        if bestScore > self.mergeThreshold:
            # update existing
            targetPattern.update(bestMatch)    
        else:
            # add new
            targetPattern = DynamicPattern(pattern, self.trailSize)
            self.patterns.append(targetPattern)

        # Update pattern-match counters (so we can calculate each pattern usage)
        for pattern in self.patterns:
            if pattern is not targetPattern:
                pattern.updateNoMatch()

        # Peridically compact the store by merging similar patterns and removing not-used patterns
        self.compactCounter += 1
        if self.compactCounter > self.compactThreshold:
            patterns = self.compact()
            self.compactCounter = 0

    def compact(self):
        result = []
        candidates = set(self.patterns)
        merged = set()
        for base in self.patterns:
            candidates.discard(base) # prevent matching/merging with itself
            if base in merged:
                # base is already merged
                continue
            if base.getMatchRate() < self.forgetThreshold:
                # forget this pattern
                continue
            for other in candidates.copy():
                match, score, transform = PatternUtility.getBestMatch(other, base)
                if score > self.mergeThreshold:
                    base.merge(other, transform)
                    candidates.discard(other)
                    merged.add(other)
            result.append(base)
        return result    

class DynamicPattern(Pattern):
    def __init__(self, pattern, trailSize = 100):
        self.screwHits = []
        self.matchRate = MatchRate(trailSize)
        self.trailSize = trailSize
        self.update(pattern.screws)

    def merge(self, another, transform):
        # Transform the other pattern to match this one
        screwsToMerge = [] 
        for screw in another.screwHits:
            screwsToMerge.append(screw.transform(transform))
        # Extract common and different screws from another
        matched = LayoutUtility.match(self.screwHits, screwsToMerge)
        additional = set(screwsToMerge)
        # merge the common screws (occurrences)
        for match in matched:
            match[0].merge(match[1])
            additional.discard(match[1])
        # add the extracted screws to the pattern
        self.screwHits.extend(additional)
        # Combine both patterns pattern matches
        self.matchRate.merge(another.matchRate)
        # Update the base pattern 
        self._updateBase()

    def update(self, layout):
        self.matchRate.next()
        # extract common and additional screws from match
        matched = LayoutUtility.match(layout, self.screwHits)
        additional = set(layout)
        # add occurrence for the common screws
        for match in matched:
            match[1].update(match[0])
            additional.remove(match[0])
        # add additional screws as new
        for screw in additional:
            self.screwHits.append(DynamicScrew(screw, self.trailSize))
        # update the base pattern 
        self._updateBase()
    
    def updateNoMatch(self):
        self.matchRate.next(False)

    def getMatchRate(self):
        return self.matchRate.getMatchRate()
    
    def _updateBase(self):
        self.screws = []
        for screw in self.screwHits:
            self.screws.append(screw)
        self._updateDistances()

class DynamicScrew(Screw):
    def __init__(self, screw, trailSize = 100):
        self.trailSize = trailSize
        self.occurences = []
        self.update(screw)

    def update(self, screw):
        self.occurences.append(screw)
        if len(self.occurences) > self.trailSize:
            self.occurences.pop(0)
        self._updateInternal()

    def updateNotFound(self):
        if len(self.occurences) > 0:
            self.occurences.pop(0)
            self._updateInternal()

    def merge(self, another):
        self.occurences = another.occurences + self.occurences
        while len(self.occurences) > self.trailSize:
            self.occurences.pop(0)
        self._updateInternal()

    def transform(self, transform):
        result = None
        for occurence in self.occurences:
            newScrew = Screw(transform.apply(occurence.center), occurence.diameter)
            if result == None:
                result = DynamicScrew(newScrew, self.trailSize)
            else:    
                result.occurences.append(newScrew)
        result._updateInternal()
        return result

    def _updateInternal(self):
        # calculate means of the center and diameter
        sumX = sumY = sumD = 0
        for screw in self.occurences:
            sumX += screw.center.x
            sumY += screw.center.y
            sumD += screw.diameter
        count = len(self.occurences)
        meanX = sumX / count
        meanY = sumY / count
        meanD = sumD / count
        # Update the base (Screw) instance
        self.center = Point(meanX, meanY)
        self.diameter = meanD

class MatchRate:
    def __init__(self, trailSize = 100):
        self.history = []
        self.tailSize = trailSize

    def next(self, match = True):
        self.history.append(match)
        if len(self.history) > self.tailSize:
            self.history.pop(0)

    def getMatchRate(self):
        count = 0.0
        for match in self.history:
            if match == True:
                count += 1.0
        return count / len(self.history)        

    def merge(self, another):
        # logical or histories
        result = []
        left = self.history.copy()
        right = another.history.copy()
        while len(left) > 0 or len(right) > 0:
            l = left.pop(0) if len(left) > 0 else False 
            r = right.pop(0) if len(right) > 0 else False
            result.append(l or r)
        self.history = result    

# -----------------------------------------------------------------------------
# Proposer implementation

class PatternMatchingProposer(ROIProposer):

    def __init__(self, layoutStore = LearningPatternStore()):
        self.layoutStore = layoutStore

    def startAssembly(self, scene):
        pass

    def finishAssembly(self, layout):
        # pattern learning logic
        self.layoutStore.addLayout(layout)

    def proposeROIs(self, layout):
        if len(layout) < 2:
            # we need at least 2 points to search for a match
            return []
        
        screws, score, transform = self.layoutStore.getBestMatch(Pattern(layout))
        if score == 0:
            return []
        
        # Remove the matched screws from the proposal to the the layout extension
        matchedPairs = LayoutUtility.match(screws, layout)
        for pair in matchedPairs:
            if pair[0] in screws: screws.remove(pair[0])
            if pair[1] in screws: screws.remove(pair[1])
        
        return LayoutUtility.getROIs(screws)
