import jetson.inference
import jetson.utils
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input_dataset", required = True, help = "path of the input video")
ap.add_argument("-n","--net", required = True, help = "pre-trained model to load (see below for options)")
ap.add_argument("-t","--threshold", required = True, help = "minimum detection threshold to use")

args = vars(ap.parse_args())

inputURL = args['input_dataset']
network = args['net']
thres = float(args['threshold'])

# load the object detection model
net = jetson.inference.detectNet(network, threshold=thres)

display = jetson.utils.glDisplay()

cap = cv2.VideoCapture(inputURL)

while(cap.isOpened()):
    ret, frame = cap.read()
    if (ret):
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        cuda_mem = jetson.utils.cudaFromNumpy(frame_rgba)
        detections = net.Detect(cuda_mem, frame.shape[1],frame.shape[0])

        for detection in detections:
            if (detection.ClassID == 10):
                top = int(detection.Top)
                left = int(detection.Left)
                bottom = int(detection.Bottom)
                right = int(detection.Right)
                frame = cv2.rectangle(frame, (right, bottom), (left, top), (255, 0, 0), 2)

        cv2.imshow("img", frame)

        # print all detections and tags
        #display.RenderOnce(cuda_mem, frame.shape[1], frame.shape[0])
        #display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        if cv2.waitKey(24) & 0xFF == ord('q'):
            break

    else:
        break

cv2.destroyAllWindows()