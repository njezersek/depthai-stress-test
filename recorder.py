#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import time
from pathlib import Path
import blobconverter

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
depth_dir = data_dir / "depth"
depth_dir.mkdir(exist_ok=True)
left_dir = data_dir / "left"
left_dir.mkdir(exist_ok=True)
right_dir = data_dir / "right"
right_dir.mkdir(exist_ok=True)

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = True
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
rgb = pipeline.create(dai.node.ColorCamera)
depth = pipeline.create(dai.node.StereoDepth)
nn = pipeline.create(dai.node.NeuralNetwork)
sys_log = pipeline.create(dai.node.SystemLogger)
xout = pipeline.create(dai.node.XLinkOut)
xout_left = pipeline.create(dai.node.XLinkOut)
xout_right = pipeline.create(dai.node.XLinkOut)
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_sys = pipeline.create(dai.node.XLinkOut)


xout.setStreamName("depth")
xout_left.setStreamName("left")
xout_right.setStreamName("right")
xout_nn.setStreamName("nn")
xout_sys.setStreamName("sysinfo")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setPreviewSize(320, 320)
rgb.setInterleaved(False)

nn.setBlobPath(str(blobconverter.from_zoo(name="yolop_320x320", zoo_type="depthai", shaves = 7)))
nn.setNumPoolFrames(4)
nn.input.setBlocking(False)
nn.setNumInferenceThreads(2)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(xout.input)
depth.rectifiedLeft.link(xout_left.input)
depth.rectifiedRight.link(xout_right.input)
sys_log.out.link(xout_sys.input)


rgb.preview.link(nn.input)
nn.out.link(xout_nn.input)

last_capture = time.time()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

	# Output queue will be used to get the disparity frames from the outputs defined above
	q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
	q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
	q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)
	q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
	q_sys = device.getOutputQueue(name="sysinfo", maxSize=4, blocking=False)

	while True:
		in_depth = q_depth.get()  # blocking call, will wait until a new data has arrived
		in_left = q_left.get()
		in_right = q_right.get()
		in_nn = q_nn.get()
		in_sys = q_sys.get()
		depth_frame = in_depth.getFrame()
		left_frame = in_left.getCvFrame()
		right_frame = in_right.getCvFrame()

		depth_vis = (depth_frame.copy().astype(np.float64) / 5000 * 256)
		depth_vis = np.clip(depth_vis, 0, 255).astype(np.uint8)
		depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_INFERNO)

		cv2.imshow("depth_color", depth_vis)
		cv2.imshow("left", left_frame)
		cv2.imshow("right", right_frame)

		key = cv2.waitKey(1)

		if key == ord("q"):
			break
		
		t = time.time()
		if t - last_capture > 10*60:
			np.save(f"{depth_dir}/depth_{t}.npy", depth_frame)
			cv2.imwrite(f"{left_dir}/left_{t}.png", left_frame)
			cv2.imwrite(f"{right_dir}/right_{t}.png", right_frame)

			with open(f"{data_dir}/temperature.txt", "a") as f:
				f.write(f"{t}, {in_sys.chipTemperature.average}\n")
			
			last_capture = time.time()