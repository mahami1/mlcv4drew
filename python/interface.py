# -----------------------------------------------------------------------------
# Description: Defined the interfaces of the main components
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports:

from abc import ABC, abstractmethod

# -----------------------------------------------------------------------------
# Definitions:

# 
# Description: The base interface to all components
#
class Module(ABC):
	
    #
    # Description: Called at ethe begining of new scene processing.
    # Arguments:
    #   scene - the image of the disassembly scene
    # Returns: nothing
    #
    @abstractmethod
    def startAssembly(self, scene):
        pass
   
    #
    # Description: Called at the end of the scene processing, 
    #   to give feedback to the components
    # Arguments:
    #   scene - list of detected Screw instances
    # Returns: nothing
    #
    @abstractmethod
    def finishAssembly(self, layout):
        pass

# 
# Description: Performs the initial detection of the regions of interest (ROI)
#
class ROIExtractor (Module):
    #
    # Description: extracts the regions of interest (ROIs) from the scene. 
    # Arguments: none
    # Returns: none
    #
    @abstractmethod
    def extractROIs(self):
        pass

# 
# Description: Proposes new regions of interest based on the already detected screw layout.
#
class ROIProposer(Module):
    
    # Description: Proposes new regions of interest based on the already detected screw layout.
    # Arguments:
    #   layout - list of Screw instances
    # Returns: list of ROI rectangles
    #
    @abstractmethod
    def proposeROIs(self, layout):
	    pass

# 
# Description: Detects is screws are present in the ROIs
#
class ScrewDetector (Module):

    #
    # Description: detects screws in the proposed regions of intrerest (ROIs)
    # Arguments:
    #   rois - list of ROI rectangles
    # Returns: list of detected Screw instances
    #
    @abstractmethod
    def detectScrews(self, rois): 
        pass 

# 
# Description: Obserevr of the two stage detection algorithm.
#
class Observer(Module):
    
    #
    # Description: Stage 1 : initial ROIs extracted from the image 
    # Arguments:
    #   rois - list of ROI rectangles
    # Returns: none
    #
    @abstractmethod
    def setInitialROIs(self, rois):
        pass
    
    #
    # Description: Stage 1 : Screws detected in the the initial ROIs 
    # Arguments:
    #   layout - list of Screw instances
    # Returns: none
    #
    @abstractmethod
    def setInitialLayout(self, layout):
        pass
    
    #
    # Description: Stage 2 : Proposed new ROIs based on the nitial Screw layout 
    # Arguments:
    #   rois - list of proposed extra ROI rectangles
    # Returns: none
    #
    @abstractmethod
    def setProposedROIs(self, rois):
        pass
    
    #
    # Description: Stage 2 : The final screw layout combining outputs of stage 1 and 2
    # Arguments:
    #   lauyout - final layout - list of Screw instances
    # Returns: none
    #
    @abstractmethod
    def setLayoutExtension(self, layout):
        pass

# 
# Description: Represent a sata source for assembly scenes.
#
class SceneSource:

    # Description: 
    # Arguments: none
    # Returns: True if nextScene() call will return valid scene
    #
    @abstractmethod
    def hasNext(self):
        pass

    # Description: 
    # Arguments: none
    # Returns: a Scene instance or None if end of data
    #
    @abstractmethod
    def nextScene(self):
        pass
