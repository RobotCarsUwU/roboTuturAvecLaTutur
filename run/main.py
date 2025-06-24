##
## EPITECH PROJECT, 2025
## Main
## File description:
## Main
##

import tensorflow as tf
from UNet import UNetDetector
import os
import depthai as dai
import cv2
from raycast import raycast

def main():
    model_path = "./unet_simple.weights.h5"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"Gpus are: {gpus}")

    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    detector = UNetDetector()

    detector.load(model_path)
    print("Model loaded")


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
            result = detector.predict(frame)
            distances = raycast(result, n=48)
            for i, d in enumerate(distances):
                print(f"Ray {i}: {d}")
            cv2.imshow("RGB Preview", result)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite("snapshot.png", frame)
                print("saved in snapshot.png")
            elif key == ord('q'):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
