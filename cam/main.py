import cv2
import depthai as dai

# 1. Create pipeline
pipeline = dai.Pipeline()

# 2. Define nodes
# 2.1 Color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)  # the color sensor
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# We output a 640×360 preview to keep USB bandwidth low
camRgb.setPreviewSize(640, 360)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
# Link camera preview → USB
camRgb.preview.link(xoutRgb.input)

# 2.2 (Optional) Left/right mono for stereo/depth
monoLeft  = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

xoutLeft  = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")
monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

# 3. Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Retrieve the output queues
    qRgb   = device.getOutputQueue(name="rgb",   maxSize=4, blocking=False)
    qLeft  = device.getOutputQueue(name="left",  maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    print("Streaming... press 'q' to exit.")
    while True:
        inRgb   = qRgb.get()    # blocking by default
        frameRgb = inRgb.getCvFrame()  # numpy array, BGR format

        inLeft  = qLeft.get()
        frameL  = inLeft.getCvFrame()  # grayscale
        inRight = qRight.get()
        frameR  = inRight.getCvFrame()

        # Show
        cv2.imshow("RGB",   frameRgb)
        cv2.imshow("Mono L", frameL)
        cv2.imshow("Mono R", frameR)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
