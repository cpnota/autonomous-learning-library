import numpy as np

def to_grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)

def downsample(frame):
    return frame[::2, ::2]

def to_torch(frame):
    # input: (H, W)
    # output: (C, H, W)
    return np.expand_dims(frame, axis=0)
