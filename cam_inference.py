import cv2
import numpy as np
import depthai as dai
import time


NN_WIDTH, NN_HEIGHT = 300, 300
VIDEO_WIDTH, VIDEO_HEIGHT = 750, 750


# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a neural network
# detection_nn = pipeline.create(dai.node.NeuralNetwork)
# detection_nn.setBlobPath('exported/mobilenetv2_test.blob')
# detection_nn.input.setBlocking(False)

detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setBlobPath('exported/mobilenetv2_test.blob')
detection_nn.setNumInferenceThreads(2)
detection_nn.input.setBlocking(False)

# Define camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(VIDEO_WIDTH, VIDEO_HEIGHT)
cam.setInterleaved(False)
cam.setFps(60)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Define manips
manip_rgb = pipeline.create(dai.node.ImageManip)
manip_rgb.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip_rgb.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
manip_rgb.inputConfig.setWaitForMessage(False)

# Create outputs
xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("cam")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")

cam.preview.link(manip_rgb.inputImage)
cam.preview.link(xout_cam.input)
manip_rgb.out.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue("cam", 4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False

    while True:
        in_frame = q_cam.get()
        in_nn = q_nn.get()
        frame = in_frame.getCvFrame()

        # output = np.array(in_nn.getLayerFp16(name='output')).reshape((1, 1, -1, 7))
        print(in_nn.detections)

        # show fps
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, (color_black))

        # show frame
        cv2.imshow("Detections", frame)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)
            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break