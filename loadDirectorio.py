import jetson.inference
import jetson.utils
import os
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input_dataset", required = True, help = "path of the input dataset")
ap.add_argument("-n","--net", required = True, help = "pre-trained model to load (see below for options)")
ap.add_argument("-t","--threshold", required = True, help = "minimum detection threshold to use")

args = vars(ap.parse_args())

inputURL = args['input_dataset']
network = args['net']
thres = float(args['threshold'])

# load the object detection model
net = jetson.inference.detectNet(network, threshold=thres)

display = jetson.utils.glDisplay()
data = os.listdir(inputURL)
data = np.sort(data)

for img in data:
    frame = cv2.imread(os.path.join(inputURL,img))
    imagen, width, height = jetson.utils.loadImageRGBA(os.path.join(inputURL,img))

    detections = net.Detect(imagen, width, height)

    # print only detected traffic lights
    for detection in detections:
        if (detection.ClassID == 10):
            top = int(detection.Top)
            left = int(detection.Left)
            bottom = int(detection.Bottom)
            right = int(detection.Right)
            frame = cv2.rectangle(frame, (right, bottom), (left, top), (255, 0, 0), 2)

    cv2.imshow("img", frame)

    # print all detections and tags
    #display.RenderOnce(imagen, width, height)
    #display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))


    if cv2.waitKey(24) & 0xFF == ord('q'):
        break