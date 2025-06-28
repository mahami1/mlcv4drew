# -----------------------------------------------------------------------------
# Description: Common data structures
# Author: Mihail Georgiev

# -----------------------------------------------------------------------------
# Imports:

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Definitions:

class Scene:
    # Provide labels if running in performace testing mode (where we know the 
    # locations of the screws)
    def __init__(self, image, labels = None):
        self.image = image
        self.labels = labels

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        else:
            return False        
        
    def __hash__(self):
        return int(self.x + self.y)

    def getDistance(self, otherPoint):
        dx = self.x - otherPoint.x
        dy = self.y - otherPoint.y
        d = np.sqrt(dx*dx + dy*dy)
        return int(np.round(d))

class Interval:
    def __init__(self, end1, end2):
        self.points = [end1, end2]
        self.length = None
        self.incline = None
        self.middle = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.points == other.points
        else:
            return False        

    def getLength(self):
        if self.length == None:
            dx = self.points[0].x - self.points[1].x 
            dy = self.points[0].y - self.points[1].y
            self.length = np.sqrt(dx*dx + dy*dy)
        return self.length

    def getIncline(self):
        if self.incline == None:
            dy = self.points[1].y - self.points[0].y
            self.incline = np.arcsin(dy / self.getLength()); 
            if self.points[1].x < self.points[0].x :
                # cos < 0
                self.incline = np.pi - self.incline
        return self.incline
    
    def getMiddle(self):
        if self.middle == None:
            mx = (self.points[0].x + self.points[1].x) // 2
            my = (self.points[0].y + self.points[1].y) // 2
            self.middle = Point(mx, my)
        return self.middle

class Rectangle:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.left == other.left \
                and self.top == other.top \
                and self.width == other.width \
                and self.height == other.height
        else:
            return False        

class Screw:
    
    def __init__(self, center, diameter, missing = False):
        self.missing = missing
        self.center = center
        self.diameter = diameter 

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.center == other.center
        else:
            return False        

    def __hash__(self):
        return int(self.center.x + self.center.y)

    def fromBBox(bbox):
        diameter = min(bbox.width, bbox.height)
        radius = diameter // 2
        center = Point(bbox.left + radius, bbox.top + radius)
        return Screw(center, diameter)

    def getBBox(self):
        x = np.round(self.center.x - self.diameter / 2.0)
        y = np.round(self.center.y - self.diameter / 2.0)
        return Rectangle(int(x), int(y), self.diameter, self.diameter)
	
    def getDistance(self, otherScrew):
        return self.center.getDistance(otherScrew.center)

class Transform2D:
    def __init__(self, parent = None):
        self.parent = parent
        self.matrix = [[ 1, 0, 0 ], [ 0, 1, 0 ]] # Identity transform

    def translate(self, dx, dy):
        result = Transform2D(self)
        result.matrix[0][2] += dx
        result.matrix[1][2] += dy
        return result

    def rotateDegree(self, degrees):
        return self.rotateRadians(degrees * (np.pi/180))
    
    def rotateRadians(self, radians):
        result = Transform2D(self)

	    # https://en.wikipedia.org/wiki/Rotation_matrix
		
        cos = np.cos(radians)
        sin = np.sin(radians)
    
        result.matrix[0][0] = cos
        result.matrix[0][1] = -sin
		
        result.matrix[1][0] = sin
        result.matrix[1][1] = cos
		
        return result

    def rotateDegreeAroundPoin(self, center, degrees):
        result = Transform2D(self)
        result.matrix = cv2.getRotationMatrix2D((center.x, center.y), degrees, 1)
        return result
        # Debug:
        # return self.rotateRadiansAroundPoin(degrees * (np.pi / 180.0))

    def rotateRadiansAroundPoin(self, center, radians):
        degrees = (radians * 180) / np.pi
        return self.rotateDegreeAroundPoin(center, -degrees)
        # Debug:
        # result = Transform2D(self)
        # result = result.translate(-center.x, -center.y)
        # result = result.rotateRadians(radians)
        # result = result.translate(center.x, center.y)
        # return result

    def apply(self, point):
        if self.parent != None:
            point = self.parent.apply(point)

        x = np.round(self.matrix[0][0] * point.x + self.matrix[0][1] * point.y + self.matrix[0][2])
        y = np.round(self.matrix[1][0] * point.x + self.matrix[1][1] * point.y + self.matrix[1][2])
		
        return Point(int(x), int(y))

class Pattern:

    def __init__(self, layout):
        self.screws = layout
        self._updateDistances()

    def _updateDistances(self):
        self.distances = []
        for i in range(len(self.screws)):
            self.distances.append(np.zeros(len(self.screws), dtype=int))

        for i in range(len(self.screws)):
            for j in range(i + 1, len(self.screws)):
                self.distances[i][j] = self.screws[i].getDistance(self.screws[j])
                self.distances[j][i] = self.distances[i][j]

    def getUniqueDistances(self, tolerance = 1.0):
        unique = np.unique(self.distances);  
        rounded = []
        for d in unique:
            r = int(np.round(d / tolerance) * tolerance)
            if r > 0:
                rounded.append(r)
        result = np.unique(rounded)
        result = np.sort(result)
        return result

    def getMatchingIntervals(self, distance, tolerance, inverse = False):
        result = []
        for i in range(len(self.screws)):
            for j in (range(len(self.screws)) if inverse else range(i + 1, len(self.screws))):
                if i != j and self.distances[i][j] >= (distance - tolerance) and \
                        self.distances[i][j] <= (distance + tolerance): 
                    result.append(Interval(self.screws[i].center, self.screws[j].center)) 
        return result

    def size(self):
        return len(self.screws)
