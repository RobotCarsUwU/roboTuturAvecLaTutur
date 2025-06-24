import cv2
import math

def raycast(image_path_or_array, n=30):
    if isinstance(image_path_or_array, str):
        g = cv2.imread(image_path_or_array, 0)
        if g is None: 
            return []
    else:
        if len(image_path_or_array.shape) == 3:
            g = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
        else:
            g = image_path_or_array
    
    h, w = g.shape
    ox, oy = w//2, h-1
    maxd = math.hypot(w, h)
    
    distances = []
    
    for i in range(n):
        a = -math.pi + i * math.pi / (n-1)
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
        
        distances.append(d)
    
    return distances
