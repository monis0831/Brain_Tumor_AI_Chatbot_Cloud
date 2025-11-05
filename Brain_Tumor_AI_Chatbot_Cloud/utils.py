\
import numpy as np

def simple_saliency(arr):
    """
    Lightweight saliency-style map using gradients (Sobel-like) on grayscale image.
    arr: HxWx3 float32 [0,1]
    returns HxW float32 [0,1]
    """
    import cv2
    gray = cv2.cvtColor((arr*255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag
