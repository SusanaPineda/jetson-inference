import jetson.inference
import jetson.utils

import argparse
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-n","--net", default="ssd-mobilenet-v2", help = "pre-trained model to load (see below for options)")
ap.add_argument("-t","--threshold", default=0.5, help = "minimum detection threshold to use")
ap.add_argument("-w","--width", default=1280, help = "desired width of camera stream (default is 1280 pixels)")
ap.add_argument("-he","--height", default=720, help = "desired height of camera stream (default is 720 pixels)")
ap.add_argument("-c","--camera", default="0", help = "index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")

args = vars(ap.parse_args())

network = args['net']
thres = float(args['threshold'])
width = int(args['width'])
height = int(args['height'])
camera = args['camera']

# load the object detection network
net = jetson.inference.detectNet(network, thres)

# create the camera and display
camera = jetson.utils.gstCamera(width, height, camera)
display = jetson.utils.glDisplay()

# process frames until user exits
while display.IsOpen():
    # capture the image
    img, width, height = camera.CaptureRGBA(zeroCopy=1)

    frame = jetson.utils.cudaToNumpy(img,  width,  height,  4)
    frame = frame/255
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect objects in the image (with overlay)
    detections = net.Detect(img, width, height)

    # print only detected traffic lights
    for detection in detections:
        #if (detection.ClassID == 10):
        print(detection.ClassID)
        top = int(detection.Top)
        left = int(detection.Left)
        bottom = int(detection.Bottom)
        right = int(detection.Right)
        frame = cv2.rectangle(frame, (right, bottom), (left, top), (255, 0, 0), 2)

    cv2.imshow("img",frame)

    # render the image
    #display.RenderOnce(img, width, height)

    # update the title bar
    #display.SetTitle("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))

    # print out performance info
    #net.PrintProfilerTimes()

    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()