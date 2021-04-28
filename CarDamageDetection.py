#!pip install keras==2.1.5
#!pip install tensorflow==1.14.0

import os
import sys
import random
import math
import re
import time
import json
import argparse
import cv2
import numpy as np
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import datetime


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

directory = '/content/drive/MyDrive/Car Damages/car_image_data'
#weight_inits = "coco" #imagenet, coco, or last
#target_list = ["Bumper damage","Bent frames","cracked windshield","Scratches","Scrapes","Dents","Paint scratch","Dings","cracks"]

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

    

    
class CarDamages(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "DamageType"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + target_list shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #IMAGE_MIN_DIM = 128
    #IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class CarDamagesDataset(utils.Dataset):

    def load_Car_Damage_images(self, dataset_dir, subset):#, target_list):
        # Add classes. 
        #for id_, name in enumerate(target_list):
        self.add_class("DamageType",1,"DamageType") #self.add_class("Damage-Type",id_+1, name)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations1.values()) # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]



            #print(f'This is Height: {height}')
            #print(f"this is image id: {a['filename'].split('.')[0]}")


            self.add_image(
            "DamageType",
            image_id=a['filename'].split('.')[0], # use file name as a unique image id
            path=image_path,
            width=width, height=height,
            polygons=polygons)



def load_mask(self, image_id):
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a balloon dataset image, delegate to parent class.
    image_info = self.image_info[image_id]
    if image_info["source"] != "DamageType":
        return super(self.__class__, self).load_mask(image_id)

    # Convert polygons to a bitmap mask of shape
    # [height, width, instance_count]
    info = self.image_info[image_id]
    mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                    dtype=np.uint8)
    for i, p in enumerate(info["polygons"]):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        mask[rr, cc, i] = 1

    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID only, we return an array of 1s
    return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)



def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    if info["source"] == "DamageType":
        return info["path"]
    else:
        super(self.__class__, self).image_reference(image_id)
        
def train_validate(directory):
    """Train the model."""
    # Training dataset.
    dataset_train = CarDamagesDataset()
    dataset_train.load_Car_Damage_images(directory, "train")#, target_list) #here
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarDamagesDataset()
    dataset_val.load_Car_Damage_images(directory, "val")#, target_list) # here
    dataset_val.prepare()

    return dataset_train, dataset_val


def Create_model(weight_inits):
    model = modellib.MaskRCNN(mode="training", config=config,  model_dir=MODEL_DIR)

    if weight_inits == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif weight_inits == "coco":
        # Load weights trained on MS COCO, but skip layers that are different due to the different number of classes
        model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

    elif weight_inits == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True) 

    return model


def MASK(image, mask):
    """Apply mask effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        image_mask = np.where(mask, image, gray).astype(np.uint8)
    else:
        image_mask = gray
    return image_mask


def detect_and_mask(model, image_path=None):
    assert image_path
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

                                   
    # Image ?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Mask image
        masked = MASK(image, r['masks'])
        # Save output
        file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, masked)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>",help="'train' or 'mask'")
    parser.add_argument("--dataset", required=False,metavar="/path/to/dataset", help=" Directory of dataset")
    parser.add_argument("--weights", required=True,metavar=" paths to .h5 file or  'coco' ", help=" Directory of dataset")
    parser.add_argument('--image', required=False, metavar="path or URL to image", help='Image to apply the mask effect on')            
    args = parser.parse_args()


    # validate the required arguments.

    if args.command == "train":
        assert args.dataset, '--dataset is needed, add the path to the directory.'
    elif args.command == "mask":
        assert args.image, '--image is required for masking.'

    # Create Configurations.

    if args.command == "train":
        config = CarDamages()
    
    else:
        class InferenceConfig(CarDamages):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create dataset

    if args.command == 'train':
        directory = args.dataset
        train, val = train_validate(directory)


    # Create Model

    if args.command == 'train':
        if args.weights.lower() == "coco":
            model = Create_model(weight_inits = 'coco')

        elif args.weights.lower() == "imagenet":
            model = Create_model(weight_inits = "imagenet")

        elif args.weights.lower() == "last":
            model = Create_model(weight_inits = "last")
        
        else:
            model = Create_model(weight_inits = args.weights.lower())
            


    # train model
    if args.command == 'train':
        model.train(train, val, learning_rate=config.LEARNING_RATE, epochs=3, layers='heads')

        print(' ')
        print(' DONE! ')

    # Test Model

    if args.command == 'mask':
        detect_and_mask(model, image_path=args.image)




    
    
