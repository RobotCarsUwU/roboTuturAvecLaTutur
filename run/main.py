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
import numpy as np
import json
from tensorflow import keras
from Ai.NeuralNetwork.MLP import MLP

def main():
    model_path = "./unet_simple.weights.h5"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"Gpus are: {gpus}")

    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    detector = UNetDetector()

    detector.load(model_path)
    print("Model loaded")

    try:
        model = keras.models.load_model('./racing_model.keras', custom_objects={'MLP': MLP})
        print("AI Model loaded")
    except:
        print("Unable to load keras model")
        return
    
    try:
        with open('simple_stats.json', 'r') as f:
            stats = json.load(f)
        min_vals = np.array(stats['min'])
        range_vals = np.array(stats['range'])
        print("Stats loaded")
    except:
        print("Unable to load stats")
        return

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
        step_count = 0

        while True:
            frame = qRgb.get().getCvFrame()
            result = detector.predict(frame)
            distances = raycast(result, n=50)
            raycast_normalized = (distances - min_vals) / range_vals
            predictions = model(raycast_normalized, training=False).numpy()

            speed = np.clip(predictions[0][0], 0.0, 0.8)
            steering = np.clip(predictions[0][1], -0.8, 0.8)

            if step_count % 50 == 0:
                print(f"Step {step_count}, Speed: {speed:.3f}, Steering: {steering:.3f}")

            step_count += 1

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
