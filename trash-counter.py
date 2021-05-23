#import packages
import os
import numpy as np
import cv2
import argparse
import dlib
import imutils
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import FPS


# contruct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y","--yolo", required=True,
    help="path to YOLO 'cfg' file")
ap.add_argument("-w", "--weight", required=True,
    help="path to YOLO pre-trained weights file")
ap.add_argument("-l","--lable", required=True,
    help="path to class lable file")
ap.add_argument("-i", "--input", type=str,
    help="path to input video file")
ap.add_argument("-o", "--output", type=str, 
    help="path to output file")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
ap.add_argument("--device", default='cpu', help="Device to be use either 'cpu' or 'gpu'")
args = vars(ap.parse_args())

# load YOLO class lable
lablesPath = os.path.join(args["lable"])
CLASSES = open(lablesPath).read().strip().split("\n")

#load YOLO  cfg and weight
print("[INFO] loading cfg and weights files...")
net = cv2.dnn.readNetFromDarknet(args["yolo"], args["weight"])

# check if using GPU
if(args.get('cpu')):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print('Using CPU device.')
elif(args.get('gpu')):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print('Using GPU device.')

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# get video source (Video Stream or Video file)
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])
    
# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0

# set counts as zero (initial)
counts = 0

fps = FPS().start()

# Video source propeties manipulation
while True:

    ret, frame = vs.read()
    #frame = frame[1] if args.get("input", False) else frame
    frame = cv2.resize(frame,(416,416), cv2.INTER_LINEAR)

    if args["input"] is not None and frame is None:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
	    (H, W) = frame.shape[:2]

    # define codec and VideoWriter for if there is output
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H))

    status = "Waiting"
    rects = []

    if totalFrames % args["skip_frames"] == 0:

        status = "Detecting"
        trackers = []
    
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (W, H), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(getOutputsNames(net))

        # convert frame to blob to be put as input for network
    
        bboxes =[]
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                if confidence > args["confidence"]:
                    bbox = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = bbox.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height /2))

                    right = width + x -1
                    bottom = height + y -1

                    bboxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x, y, right, bottom)
                    tracker.start_track(rgb,rect)

                    trackers.append(tracker)

        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX,startY,endX, endY))

        # counting mechanisms

        cv2.line(frame, (0,H // 2), (W, H // 2), (0, 255, 255), 2)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                # y-coordinate of current centroid - mean of previous centroid
                # equal to direction of object moving (-ve up) (+ve down)
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        counts += 1
                        to.counted = True
                    
                    elif direction > 0 and centroid[1] > H // 2:
                        counts += 1
                        to.counted = True

                    #y == (H // 2)
                    #counts += 1
                    #to.counted = True
            
            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0]-10, centroid[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info = [("Status", status)]

        for (i, (k, v)) in enumerate(info):
            text = "{} : {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(frame,(f"Trash detected: {counts}".format(counts)), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)

        if writer is not None:
            writer.write(frame)
        
        cv2.imshow ('Input', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #totalFrames += 1
    #fps.update()

if writer is not None:
    writer.release()

if not args.get("input", False):
    vs.stop()
else:
    vs.release()

# When everything done, release the capture

cv2.destroyAllWindows()
