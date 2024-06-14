import ctypes
import numpy as np
from numpy import rot90, ndarray, newaxis, uint8, zeros
from numpy.ctypeslib import as_array
from .base import Processor


class NumpyProcessor(Processor):
    def __init__(self, color_mode):
        self.cvtcolor = None
        self.color_mode = color_mode
        self.PBYTE = ctypes.POINTER(ctypes.c_ubyte)
        if self.color_mode=='BGRA':
            self.color_mode = None

    def process_cvtcolor(self, image):
        import cv2

        # only one time process
        if self.cvtcolor is None:
            color_mapping = {
                "RGB": cv2.COLOR_BGRA2RGB,
                "RGBA": cv2.COLOR_BGRA2RGBA,
                "BGR": cv2.COLOR_BGRA2BGR,
                "HSV": cv2.COLOR_BGR2HSV,
                "GRAY": cv2.COLOR_BGRA2GRAY
            }
            cv2_code = color_mapping[self.color_mode]
            if cv2_code != cv2.COLOR_BGRA2GRAY:
                self.cvtcolor = lambda image: cv2.cvtColor(image, cv2_code)
            else:
                self.cvtcolor = lambda image: cv2.cvtColor(image, cv2_code)[
                    ..., np.newaxis
                ] 
        return self.cvtcolor(image)

    def shot(self, image_ptr, rect, width, height):
        ctypes.memmove(image_ptr, rect.pBits, height*width*4)


    def processA0(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = region[1] * pitch
        height = region[3] - region[1]
        size = pitch * height
        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        image = np.ndarray((height, pitch // 4, 4), dtype=np.uint8, buffer=buffer)
        #image = image[:, :width, :]        
        if region[3] - region[1] != image.shape[0]:
            image = image[region[1]:region[3], :, :]
        if region[2] - region[0] != image.shape[1]:
            image = image[:, region[0]:region[2], :]
            
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image
    
    def processA90(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = (width - region[2]) * pitch
        width = region[2] - region[0]
        size = pitch * width
        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        image = np.ndarray((width, pitch // 4, 4), dtype=np.uint8, buffer=buffer)
        image = np.rot90(image, axes=(1, 0))
        if width != image.shape[0]:
            image = image[:width, :, :]
        if height != image.shape[1]:
            image = image[:, :height, :]
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image
    
    def processA180(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = (height - region[3]) * pitch
        height = region[3] - region[1]
        size = pitch * height
        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        image = np.ndarray((height, pitch // 4, 4), dtype=np.uint8, buffer=buffer)
        image = np.rot90(image, k=2, axes=(0, 1))
        if region[3] - region[1] != image.shape[0]:
            image = image[region[1]:region[3], :, :]
        if region[2] - region[0] != image.shape[1]:
            image = image[:, region[0]:region[2], :]
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image
    
    def processA270(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = region[0] * pitch
        width = region[2] - region[0]
        size = pitch * width
        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        image = np.ndarray((width, pitch // 4, 4), dtype=np.uint8, buffer=buffer)
        image = np.rot90(image, axes=(0, 1))
        if width != image.shape[0]:
            image = image[:width, :, :]
        if height != image.shape[1]:
            image = image[:, :height, :]
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image











