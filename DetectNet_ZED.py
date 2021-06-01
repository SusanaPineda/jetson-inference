import jetson.inference
import jetson.utils
from jetson.utils import cudaFromNumpy as cfn, saveImageRGBA
import pyzed.sl as sl

import argparse
import sys
import cv2
from time import time

network = "ssd-mobilenet-v2"
overlay = "box,labels,conf"
threshold = 0.5
width = 1920
height = 1080
	
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

# load the recognition network
net = jetson.inference.detectNet(network, sys.argv, threshold)

# process frames until user exits
while cv2.waitKey(1) != ord("q"):
	err = zed.grab(runtime)
	if err == sl.ERROR_CODE.SUCCESS:
		# capture the image
		start_time = time()
		zed.retrieve_image(image_zed, sl.VIEW.LEFT)
		image_data = image_zed.get_data()
		cuda_mem = cfn(image_data)

		det_time = time()
		detections = net.Detect(cuda_mem, width, height, overlay)
		elapsed_time = time() - start_time
		d_time = time() - det_time
		print("deteccion: " + str(elapsed_time))
		print("captura y deteccion: "+str(elapsed_time))
		# print the detections
		#print("detected {:d} objects in image".format(len(detections)))

		for detection in detections:
			top = int(detection.Top)
			left = int(detection.Left)
			bottom = int(detection.Bottom)
			right = int(detection.Right)
			image_data = cv2.rectangle(image_data, (right, bottom), (left, top), (255, 0, 0), 2)
			#print(detection)
		
		cv2.imshow("img", image_data)
		#print(net.GetNetworkFPS())

		# print out performance info
		#net.PrintProfilerTimes()