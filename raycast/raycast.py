import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import sys, cv2, math

path = './raycast/masks/mask5.png'
n = 30

g = cv2.imread(path, 0)
if g is None: sys.exit()

h, w = g.shape
ox, oy = w//2, h-1
maxd = math.hypot(w, h)

for i in range(n):
    a  = -math.pi + i * math.pi / (n-1)
    dx, dy = math.cos(a), math.sin(a)
    d = None
    for t in range(int(maxd)):
        x = int(ox + dx*t)
        y = int(oy + dy*t)
        if x<0 or x>=w or y<0 or y>=h:
            break
        if g[y, x] > 128:
            d = t
            break
    print(d)
