from enum import Enum

class ProcessorBackends(Enum):
    NUMPY = 'numpy'
    CUPY = 'cupy'
    TORCHCUDA='TORCH-CUDA'

class Processor:
    def __init__(self, nvidia_gpu=False,torch_cuda=False, output_color="RGB", rotation_angle=0):
        if nvidia_gpu:
            backend = ProcessorBackends.CUPY
        elif torch_cuda:
            backend = ProcessorBackends.TORCHCUDA
        else:
            backend = ProcessorBackends.NUMPY

        self.color_mode = output_color
        self.backend = self._initialize_backend(backend)

        if nvidia_gpu or torch_cuda :
            self.set_rotation_CPfunction(rotation_angle)
        else:
            self.set_rotation_function(rotation_angle)

    def set_rotation_function(self, angle):
        angle_to_function = {
            0: self.backend.processA0,
            90: self.backend.processA90,
            180: self.backend.processA180,
            270: self.backend.processA270
        }
        self.process = angle_to_function.get(angle)
        if not self.process:
            raise ValueError(f"Unsupported rotation angle: {angle}")

    def set_rotation_CPfunction(self, angle):
        angle_to_function = {
            0: self.backend.processCPA0,
            90: self.backend.processCPA90,
            180: self.backend.processCPA180,
            270: self.backend.processCPA270
        }
        self.process = angle_to_function.get(angle)
        if not self.process:
            raise ValueError(f"Unsupported rotation angle: {angle}")

    def process(self, rect, width, height, region):
        pass

    def _initialize_backend(self, backend):
        print(f"Initializing backend: {backend}")
        if backend == ProcessorBackends.NUMPY:
            from bettercam.processor.numpy_processor import NumpyProcessor
            return NumpyProcessor(self.color_mode)
        elif backend == ProcessorBackends.CUPY:
            from bettercam.processor.cupy_processor import CupyProcessor
            return CupyProcessor(self.color_mode)
        elif backend == ProcessorBackends.TORCHCUDA:
            from bettercam.processor.TorchCuda_Processor import TorchProcessor
            return TorchProcessor(self.color_mode)
        else:
            raise ValueError(f"Unknown backend: {backend}")




