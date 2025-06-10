#!/usr/bin/env python3
import os, sys

import depthai as dai
import cv2

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setPreviewSize(640, 360)
camRgb.setPreviewKeepAspectRatio(True)
camRgb.setFps(35)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

with dai.Device(pipeline) as dev:
    qRgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
    while True:
        frame = qRgb.get().getCvFrame()
        cv2.imshow("RGB Preview", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite("snapshot.png", frame)
            print("saved in snapshot.png")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
