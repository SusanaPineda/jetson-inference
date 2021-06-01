import jetson.inference
import jetson.utils
from jetson.utils import cudaFromNumpy as cfn, saveImageRGBA
import pyzed.sl as sl

import argparse
import sys
import cv2

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
	
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
init_params.camera_fps = 30  # Set fps at 30


status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
	print(repr(status))
	exit(1)
	
runtime = sl.RuntimeParameters()

# Create a matrix to store the image
image_zed = sl.Mat()
width = opt.width
height = opt.height

# load the recognition network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# process frames until user exits
while cv2.waitKey(1) != ord("q"):
	err = zed.grab(runtime)
	if err == sl.ERROR_CODE.SUCCESS:
   	# capture the image
		zed.retrieve_image(image_zed, sl.VIEW.LEFT)
		image_data = image_zed.get_data()
		cuda_mem = cfn(image_data)

		detections = net.Detect(cuda_mem, width, height, opt.overlay)

		# print the detections
		print("detected {:d} objects in image".format(len(detections)))

		for detection in detections:
			top = int(detection.Top)
			left = int(detection.Left)
			bottom = int(detection.Bottom)
			right = int(detection.Right)
			image_data = cv2.rectangle(image_data, (right, bottom), (left, top), (255, 0, 0), 2)
			#print(detection)
		
		cv2.imshow("img", image_data)
		print(net.GetNetworkFPS())

		# print out performance info
		net.PrintProfilerTimes()