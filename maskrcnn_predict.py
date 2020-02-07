# USAGE
# python maskrcnn_predict.py --weights mask_rcnn_coco.h5 --labels coco_labels.txt --image images/30th_birthday.jpg

# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", required=True,
                help="path to class labels file")
# ap.add_argument("-i", "--image", required=False,
# help="path to input image to apply Mask R-CNN to")
args = vars(ap.parse_args())

start = time.time()
# load the class label names from disk, one label per line
CLASS_NAMES = open(args["labels"]).read().strip().split("\n")

# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)


class SimpleConfig(Config):
    # give the configuration a recognizable name
    NAME = "coco_inference"

    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes (we would normally add +1 for the background
    # but the background class is *already* included in the class
    # names)
    NUM_CLASSES = len(CLASS_NAMES)


# initialize the inference configuration
config = SimpleConfig()

# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image

video_path = 'Corners.mp4'
output_path = 'Corners1.avi'
fps = 15

vcapture = cv2.VideoCapture(video_path)


def save_image_every_seconds(video_path, start_sec, step_sec, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("False")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    end_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) * fps
    if start_sec > end_sec:
        print("start time > end time")
        return False

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    sec = start_sec
    while sec < end_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(fps * sec))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{:02.0f}m{:05.2f}s.{}'.format(base_path, sec // 60, sec % 60, ext), frame)
        else:
            break
        sec += step_sec
    print("Saved directory: {}".format(dir_path))
    return True


# save_image_every_seconds(video_path , 1, 0.1, 'output', 'frame')

width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # uses given video width and height
height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
vwriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))  #

num_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
print("Number of Frames: ", num_frames)
print("Original Width, Height: ", width, height)

count = 0
success = True
start = time.time()
while success:
    if count % 10 == 0:
        print("frame: ", count)

    count += 1  # see what frames you are at
    # Read next image
    success, image = vcapture.read()

    if count % 1000 == 0:
        success = False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = imutils.resize(image, width=512)

    # perform a forward pass of the network to obtain the results
    # print("[INFO] making predictions with Mask R-CNN...")
    r = model.detect([image], verbose=1)[0]

    # loop over of the detected object's bounding boxes and masks
    for i in range(0, r["rois"].shape[0]):
        # extract the class ID and mask for the current detection, then
        # grab the color to visualize the mask (in BGR format)
        classID = r["class_ids"][i]
        mask = r["masks"][:, :, i]
        color = COLORS[classID][::-1]
        # visualize the pixel-wise mask of the object
        image = visualize.apply_mask(image, mask, color, alpha=0.5)

    # convert the image back to BGR so we can use OpenCV's drawing
    # functions
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # loop over the predicted scores and class labels
    for i in range(0, len(r["scores"])):
        # extract the bounding box information, class ID, label, predicted
        # probability, and visualization color
        (startY, startX, endY, endX) = r["rois"][i]
        classID = r["class_ids"][i]
        label = CLASS_NAMES[classID]
        score = r["scores"][i]

        if score > 0.85:
            color = [int(c) for c in np.array(COLORS[classID]) * 255]

            # draw the bounding box, class label, and score of the object
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.3f}".format(label, score)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
    vwriter.write(image)

vcapture.release()
vwriter.release()

# show the output image
# cv2.imwrite("out/frame" + str(j) + ".jpg", image)

end = time.time()
print(end - start)
