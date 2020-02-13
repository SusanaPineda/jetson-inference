import jetson.inference
import jetson.utils

import argparse
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-n","--net", required = True, default="ssd-mobilenet-v2", help = " detect network")
ap.add_argument("-t","--threshold", required = True , default=0.5, help = "threshold")
ap.add_argument("-w","--width", required = True, default=1280, help = "width")
ap.add_argument("-he","--height", required = True, default=720, help = "height")
ap.add_argument("-c","--camera", required = True, default="0", help = "camera")

args = vars(ap.parse_args())

network = args['net']
thres = float(args['threshold'])
width = float(args['width'])
height = float(args['height'])
camera = args['camera']

# load the object detection network
net = jetson.inference.detectNet(network, thres)

# create the camera and display
camera = jetson.utils.gstCamera(width, height, camera)
display = jetson.utils.glDisplay()

# process frames until user exits
while display.IsOpen():
    # capture the image
    img, width, height = camera.CaptureRGBA()

    # detect objects in the image (with overlay)
    detections = net.Detect(img, width, height)

    # print only detected traffic lights
    for detection in detections:
        if (detection.ClassID == 10):
            top = int(detection.Top)
            left = int(detection.Left)
            bottom = int(detection.Bottom)
            right = int(detection.Right)
            frame = cv2.rectangle(img, (right, bottom), (left, top), (255, 0, 0), 2)

    cv2.imshow("img", frame)

    # render the image
    #display.RenderOnce(img, width, height)

    # update the title bar
    #display.SetTitle("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))

    # print out performance info
    #net.PrintProfilerTimes()

    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()