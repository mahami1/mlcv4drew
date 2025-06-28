# -----------------------------------------------------------------------------
# Description: Image / scene sources
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports

import os
import cv2
import csv
import random
import numpy as np

from imutils import paths

from interface import SceneSource
from data import Scene, Screw, Point, Rectangle

# -----------------------------------------------------------------------------
# Definitions

# 
# Description: Reads the images in the specified directory and annotations (if present). 
#
class SceneReader (SceneSource):
    
    class AnnotatedImage:
        def __init__(self, imageFile, annotationsFile = None):
            self.imageFile = imageFile
            self.annotationsFile = annotationsFile

    def __init__(self, repoDirs):
        self.repoDirs = repoDirs
        self.currentRepo = 0
        self.currentRepoContent = []
        self._loadRepo()

    def hasNext(self):
        return  len(self.currentRepoContent) > 0 or (self.currentRepo + 1) < len(self.repoDirs)

    def nextScene(self):
        result = None

        if len(self.currentRepoContent) == 0:
            if (self.currentRepo + 1 < len(self.repoDirs)):
                self.currentRepo += 1
                self._loadRepo()

        if len(self.currentRepoContent) > 0:
            annotatedImage = self.currentRepoContent.pop(0)
            image = cv2.imread(annotatedImage.imageFile)
            labels = None
            if annotatedImage.annotationsFile != None:
                labels = self._loadAnnotations(annotatedImage.annotationsFile)
            result = Scene(image, labels)

        return result

    def _loadRepo(self):
        self.currentRepoContent = []
        for imagePath in paths.list_images(self.repoDirs[self.currentRepo] + "/Images"):
            imageFile = imagePath.split('\\')[-1]
            imageFileName, imageFileExt = imageFile.split('.')
            aFile = imageFileName + ".csv"
            annotationPath = self.repoDirs[self.currentRepo]  + "/Annotations/" + aFile
            if not os.path.isfile(annotationPath):
                annotationPath = None
            self.currentRepoContent.append(SceneReader.AnnotatedImage(imagePath, annotationPath))

    def _loadAnnotations(self, file):
        result = []
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['label_name'] == "screw":
                    # Ignore holes .. for now (TODO)
                    x = int(row['bbox_x'])
                    y = int(row['bbox_y'])
                    w = int(row['bbox_width'])
                    h = int(row['bbox_height'])
                    result.append(Screw.fromBBox(Rectangle(x, y, w, h)))
        return result

class SceneGenerator (SceneSource):

    def __init__(self, sceneSource, sceneCount):
        self.sceneSource = sceneSource
        self.sceneCount = sceneCount
        self.scenes = []

    def hasNext(self):
        return self.sceneCount > 0

    def nextScene(self):
        result = None
        if self.hasNext():
            if self.sceneSource.hasNext():
                result = self.sceneSource.nextScene()
                self.scenes.append(result)
            else:
                # Get random scene from the buffer
                index = int(random.uniform(0, len(self.scenes)))
                scene = self.scenes[index]
                # Apply random translation, rotation, light and contrast chnage
                result = self._augment(scene.image, scene.labels)
            
            self.sceneCount -= 1

        return result

    def _augment(self, image, annotations = None):
        image = self._filter(image, random.uniform(0.5, 1.5), random.randint(-20, 20))
        image, annotations = self._rotate(image, random.randint(-180, 180), annotations)
        image, annotations = self._translate(image, random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), annotations)
        return Scene(image, annotations)

    def _translate(self, image, x, y, annotations=None): 
        dx = int(image.shape[1] * x)
        dy = int(image.shape[0] * y)
        transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        image = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]))
        annotations = self._transformAnnotations(annotations, transform, image.shape)
        return image, annotations

    def _rotate(self, image, angle, annotations=None):
        centerX = image.shape[1]//2
        centerY = image.shape[0]//2
        transform = cv2.getRotationMatrix2D((centerX, centerY), angle, 1)
        image = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]))
        annotations = self._transformAnnotations(annotations, transform, image.shape)
        return image, annotations

    def _filter(self, image, contrast=1, brightness=0):
        # pixel transfrmation g(x) = contrast*f(x) + brightness
        # contrast = 1.5  # 1 - 3
        # brightness = 0 # 1-100
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return image

    def _transformAnnotations(self, annotations, t, shape):
        if annotations != None:
            result = []
            for screw in annotations:
                old_x = screw.center.x
                old_y = screw.center.y
                H = shape[0]
                W = shape[1]
                # rotate the center
                new_x = int(t[0][0]*old_x + t[0][1]*old_y + t[0][2])
                new_y = int(t[1][0]*old_x + t[1][1]*old_y + t[1][2])
                # if the result is out of the picture, just ignore it
                if new_x >= 0 and new_x < W and new_y >= 0 and new_y < H:
                    result.append(Screw(Point(new_x, new_y), screw.diameter))
        return result
