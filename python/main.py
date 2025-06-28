# -----------------------------------------------------------------------------
# Description: This is the main application implementing the 2-phase screw detection algorithm 
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports

import argparse
import random

from interface import Observer
from scenes import SceneReader, SceneGenerator
from observer import PerformanceObserver, ImageDecoratorObserver, CompositeObserver 
from extractor import HoughTransformROIExtractor
from detector import XceptionCNNScrewDetector
from proposer import PatternMatchingProposer, ListPatternStore, LearningPatternStore

# -----------------------------------------------------------------------------
# Definitions

class EmptyObserver(Observer):
    def startAssembly(self, scene):
        pass
    def finishAssembly(self, layout):
        pass
    def setInitialROIs(self, rois):
        pass
    def setInitialLayout(self, layout):
        pass
    def setProposedROIs(self, rois):
        pass
    def setLayoutExtension(self, layout):
        pass

class TwoPassScrewDetector:
    def __init__(self, extractor, proposer, detector, observer = EmptyObserver()):
        self.extractor = extractor
        self.proposer = proposer
        self.detector = detector
        self.observer = observer

    def extractScrewLayout(self, scene):
        # Prepare
        self.extractor.startAssembly(scene)
        self.detector.startAssembly(scene)
        self.proposer.startAssembly(scene)
        self.observer.startAssembly(scene)
		
        # Phase 1:
        initialROIs = self.extractor.extractROIs()
        self.observer.setInitialROIs(initialROIs)
		
        initialLayout = self.detector.detectScrews(initialROIs)
        self.observer.setInitialLayout(initialLayout)
		
		# Phase2:
        proposedROIs = self.proposer.proposeROIs(initialLayout)
        self.observer.setProposedROIs(proposedROIs)
		
        layoutExtension = self.detector.detectScrews(proposedROIs)
        self.observer.setLayoutExtension(layoutExtension)
		
		# Finish
        finalLayout = initialLayout + layoutExtension
        self.extractor.finishAssembly(finalLayout)
        self.detector.finishAssembly(finalLayout)
        self.proposer.finishAssembly(finalLayout)
        self.observer.finishAssembly(finalLayout)
		
        return finalLayout

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="main.py", description="Tests the performance of the two-pass screw detection algorithm "
                                     "over specified set of labeled disassembly scenes")

    parser.add_argument("-scenes", required=True, nargs='+', # type = readable directory, 
                        help="Input scene directories. Images and labels must be in sub-directory 'Images' and 'Annotations'repectively")
    
    parser.add_argument("-output", required=False, # type = writable directory, 
                        help="Directory to store the labeled scene images showing the various stages of the two-pass algorithm processing.")
    
    parser.add_argument("-generate", required=False, type=int, help="The number of scenes to generate from the available input scenes")
    parser.add_argument("-learn", required=False, action="store_true", default=False, help="Use lerning pattern store")
    parser.add_argument("-weights", required=True, type=argparse.FileType("r"), help="The Xception CNN weights to be used by the screw detector")
    parser.add_argument("-seed", type=int, required=False, help="The random generator seed value")

    args = parser.parse_args()

    if args.seed != None:
        random.seed(args.seed)

    sceneSource = SceneReader(args.scenes)
    if args.generate != None:
        sceneSource = SceneGenerator(sceneSource, args.generate)

    observer = PerformanceObserver()
    if args.output != None:
        observer = CompositeObserver([observer, ImageDecoratorObserver(args.output)])

    extractor = HoughTransformROIExtractor()
    detector = XceptionCNNScrewDetector(args.weights.name)
    if args.learn:
        store = LearningPatternStore()
    else:    
        store = ListPatternStore()

    proposer = PatternMatchingProposer(store)
    worker = TwoPassScrewDetector(extractor, proposer, detector, observer)

    while sceneSource.hasNext():
        scene = sceneSource.nextScene()
        if args.learn == False:
            # preload the scene annotations in pattern store (to simulate memorised pattern)
            store.locked = False
            store.addLayout(scene.labels)
            store.locked = True
        layout = worker.extractScrewLayout(scene)
    
    print("Done")