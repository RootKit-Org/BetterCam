import ctypes
import torch
from .base import Processor

class TorchProcessor(Processor):
    def __init__(self, color_mode):
        if color_mode not in ['BGRA', 'BGR', 'RGB', 'GRAY']:
            raise ValueError("Unsupported color mode. Supported modes are 'BGRA', 'BGR', 'RGBA', 'RGB', and 'GRAY'.")
        self.color_mode = color_mode

    def process_cvtcolor(self, image):
        if self.color_mode == 'RGB':
            return image[:, :, [2, 1, 0]]  # BGRA to RGB
        elif self.color_mode == 'BGR':
            return image[:, :, :3]  # BGRA to BGR
        elif self.color_mode == 'GRAY':
            # BGRA to Grayscale using the luminosity method
            return 0.2989 * image[:, :, 2] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 0]
        return image

    def processCPA0(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = region[1] * pitch
        height = region[3] - region[1]
        size = pitch * height
        buffer_ptr = ctypes.addressof(rect.pBits.contents) + offset
        buffer = (ctypes.c_char * size).from_address(buffer_ptr)
        image = torch.frombuffer(buffer, dtype=torch.uint8).reshape((height, pitch // 4, 4)).cuda()

        if region[3] - region[1] != image.shape[0]:
            image = image[region[1]:region[3], :, :]
        if region[2] - region[0] != image.shape[1]:
            image = image[:, region[0]:region[2], :]
            
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image.cpu().numpy()
    
    def processCPA90(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = (width - region[2]) * pitch
        width = region[2] - region[0]
        size = pitch * width
        buffer_ptr = ctypes.addressof(rect.pBits.contents) + offset
        buffer = (ctypes.c_char * size).from_address(buffer_ptr)
        image = torch.frombuffer(buffer, dtype=torch.uint8).reshape((height, pitch // 4, 4)).cuda()
        image = torch.rot90(image, 1, (0, 1))
        if width != image.shape[0]:
            image = image[:width, :, :]
        if height != image.shape[1]:
            image = image[:, :height, :]
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image.cpu().numpy()
    
    def processCPA180(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = (height - region[3]) * pitch
        height = region[3] - region[1]
        size = pitch * height
        buffer_ptr = ctypes.addressof(rect.pBits.contents) + offset
        buffer = (ctypes.c_char * size).from_address(buffer_ptr)
        image = torch.frombuffer(buffer, dtype=torch.uint8).reshape((height, pitch // 4, 4)).cuda()
        image = torch.rot90(image, 2, (0, 1))
        if region[3] - region[1] != image.shape[0]:
            image = image[region[1]:region[3], :, :]
        if region[2] - region[0] != image.shape[1]:
            image = image[:, region[0]:region[2], :]
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image.cpu().numpy()
    
    def processCPA270(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = region[0] * pitch
        width = region[2] - region[0]
        size = pitch * width
        buffer_ptr = ctypes.addressof(rect.pBits.contents) + offset
        buffer = (ctypes.c_char * size).from_address(buffer_ptr)
        image = torch.frombuffer(buffer, dtype=torch.uint8).reshape((height, pitch // 4, 4)).cuda()
        image = torch.rot90(image, 3, (0, 1))
        if width != image.shape[0]:
            image = image[:width, :, :]
        if height != image.shape[1]:
            image = image[:, :height, :]
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image.cpu().numpy()
