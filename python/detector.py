# -----------------------------------------------------------------------------
# Description: Screw Detector implementations.
# Author: Mihail Georgiev 

# -----------------------------------------------------------------------------
# Imports

import cv2
import os
import numpy as np
import keras
import argparse

from keras.models import *
from keras.layers import *
from keras.callbacks import *

from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

from data import Scene, Screw
from interface import ScrewDetector

from imutils import paths

# -----------------------------------------------------------------------------
# Definitions

IMAGE_SIZE= [75, 75]

def getXceptionCNN(imageSize=IMAGE_SIZE, training = False):
    # build our classifier model based on pre-trained Xception CNN:
    # 1. we don't include the top (fully connected) layers of the Xception CNN
    # 2. we add a DropOut layer followed by a Dense (fully connected)
    #    layer which generates softmax class score for each class
    # 3. we compile the final model using an Adam optimizer, with a
    #    low learning rate (since we are 'fine-tuning')
    base_model = Xception(
        include_top=False,
        weights=('imagenet' if training else None),
        input_tensor=None,
        input_shape=(imageSize[0], imageSize[1], 3))
    
    if training:
        for layer in base_model.layers:
            layer.trainable = True

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5)(x)
    # and a logistic layer 
    numberOfClasses = 3 # screw, hole, other
    outputs = Dense(numberOfClasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

class XceptionCNNScrewDetector(ScrewDetector):

    def __init__(self, weightsFile, imageSize=IMAGE_SIZE):
        self.model = getXceptionCNN(imageSize, training=False)
        self.model.load_weights(weightsFile)
        self.imageSize=imageSize
    
    def detectScrews(self, rois): 

        roiImages = []
        id = 0
        for roi in rois:
            if self._isValidROI(roi):
                roiImage = self.image[int(roi.top) : int(roi.top + roi.height), int(roi.left) : int(roi.left + roi.width)]
                if self._isValidROIimage(roiImage):
                    # Debug code
                    # file = "c:/project/mlcv4drew/rep/output/rois/roi-" + str(id).zfill(3) + ".png"
                    # cv2.imwrite(file, roiImage)
                    # id += 1
                    # ---
                    roiImage = self._normalise(roiImage)
                    roiImages.append(roiImage)

        if len(roiImages) == 0:
            return []

        tensor = np.reshape(roiImages, (len(roiImages), self.imageSize[0], self.imageSize[1], 3))
        labels = self.model.predict(tensor)

        result = []
        for label, roi in zip(labels, rois):
            if self._isScrew(label):
                result.append(Screw.fromBBox(roi))

        return result

    def _isValidROI(self, roi):
        H = self.image.shape[0]
        W = self.image.shape[1]
        if roi.left < 0 or (roi.left + roi.width) > W:
            return False
        if roi.top < 0 or (roi.top + roi.height) > H:
            return False
        return True

    def _isValidROIimage(self, image):
        blackPixels = 0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                blackChanels = 0
                for z in range(image.shape[2]):
                    if image[x][y][z] <= 25:
                        blackChanels += 1
                if blackChanels == 3:
                   blackPixels += 1     

        pixelCount = image.shape[0] * image.shape[1]
        if blackPixels > pixelCount * 0.5: 
            # ignore ROIs that are more than 50% background
            return False
        return True

    def _normalise(self, img):
        image = cv2.resize(img, self.imageSize)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float") / 255.0
        # Expand dimensions as predict expect image in batches
        image = np.expand_dims(image, axis=0) 
        return image
    
    def _isScrew(self, label):
        # label is list of probabilties for each of the 3 classes (0:hole, 1:other, 2:screw)
        max = np.argmax(label)
        return max == 2

    def startAssembly(self, scene):
        if isinstance(scene, Scene):
            self.image = scene.image
        else:
            self.image = scene
   
    def finishAssembly(self, layout):
        self.image = None

# -----------------------------------------------------------------------------
def testCNN(imageDir, weights):

    model = getXceptionCNN(training = False)
    model.load_weights(weights)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_batches = test_datagen.flow_from_directory(
        imageDir,
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        shuffle=False,
        batch_size=1)

    predicted = model.predict_generator(test_batches, steps = test_batches.samples) 

    # Decode casses to one-hot
    predicted_classes = [] 
    for ohc in predicted: 
        nc = np.argmax(ohc)
        predicted_classes.append(nc)
    
    actual_classes = test_batches.classes

    class_names = ['Hole', 'Other', 'Screw']
    labels = [0,1,2]

    # Print classification report and confusion matrix
    print(classification_report(actual_classes, predicted_classes, target_names=class_names, labels=labels))
    print("confusion matrix")
    print(confusion_matrix(actual_classes, predicted_classes, labels=labels))

def trainCNN(trainImages, validateImages, weights, batchSize = 8, epochs = 15):

    weightsDir = os.path.dirname(weights)

    # prepare train (80%) and validation (20%) datasets (split 80/20)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1, height_shift_range=0.1, 
        shear_range=0.01, zoom_range=[0.9,1.25],
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.4,1.5],
        fill_mode='reflect')

    train_batches = train_datagen.flow_from_directory(
        trainImages,
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        shuffle=True,
        batch_size=batchSize)

    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_batches = valid_datagen.flow_from_directory(
        validateImages,
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        shuffle=False,
        batch_size=batchSize)

    model = getXceptionCNN(training = True)

    # Compile model
    model.compile(optimizer=keras.optimizers.legacy.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-9, amsgrad=True), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    tensorboard_callback = TensorBoard(
        log_dir=weightsDir, 
        histogram_freq=0,
        write_graph=True,
        write_images=False)
    
    save_model_callback = ModelCheckpoint(
        os.path.join(weightsDir, 'weights.{epoch:02d}.h5'),
        verbose=3,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1)

    history = model.fit_generator(
        train_batches,
        steps_per_epoch = train_batches.samples // batchSize,
        validation_data = valid_batches, 
        validation_steps = valid_batches.samples // batchSize,
        callbacks=[save_model_callback, tensorboard_callback],
        class_weight= { 0: 2.0, 1: 2.0, 2: 1.0 }, # screw (class 2) images are twice as many as the other classes
        epochs = epochs)
    
    # save final trained weight
    model.save(weights)

# -----------------------------------------------------------------------------
# Main - Test the detector performance / train the CNN

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="detector.py", description="Tests the performance (or trains the CNN) of the XceptionCNNScrewDetector "
                                     "over specified set of classified ROI images")

    parser.add_argument("-roi", required=False, # type = readable directory, 
                        help="Directory containing cropped and classified region of interest (ROI) images."
                        "Must containg subdirectories for each class (screw, hole, other)")
    
    parser.add_argument("-weights", required=True, help="The Xception CNN weights. " 
                        "If missing, the CNN will be trained and the generated wights will be stored in that file")
    
    args = parser.parse_args()

    if os.path.exists(args.weights):  
        print("Testing Xception CNN performance ...")
        testCNN(args.roi, args.weights)
    else:    
        print("Training the Xception CNN ...")
        trainImages = os.path.join(args.roi, "train")
        validateImges = os.path.join(args.roi, "validate")
        if os.path.exists(trainImages) and os.path.exists(validateImges):
            trainCNN(trainImages, validateImges, args.weights)
        else:
            print("Unable to train CNN - the ROI directory must contain 'train' and 'validate' directories, "
                  "each one containing the subdirectories for image classes (screw, hole, other))")    
    
    print("Done")

