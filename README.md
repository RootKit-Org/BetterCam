RTX 4070 + RYZEN 7 5800X 48GB RAM

SCREEN CAPTURE RESOLUTION AT 1920X1080 

TEST FROM : BetterCam/benchmarks/bettercam_capture.py


Screen Capture FPS: 640
[BetterCam] Capture benchmark with NVIDIA GPU
Elapsed time: 3.53 seconds
Frames per second: 283.38 FPS


Screen Capture FPS: 622
[BetterCam] Capture benchmark with Torch CUDA
Elapsed time: 3.36 seconds
Frames per second: 297.91 FPS


Screen Capture FPS: 657
[BetterCam] Capture benchmark without GPU acceleration
Elapsed time: 3.50 seconds
Frames per second: 286.06 FPS

Comparison Results:
NVIDIA GPU - Elapsed time: 3.53 seconds, FPS: 283.38
Torch CUDA - Elapsed time: 3.36 seconds, FPS: 297.91
No GPU - Elapsed time: 3.50 seconds, FPS: 286.06

camera = bettercam.create(output_idx=0, output_color="BGRA", nvidia_gpu=True) TO USE DIRECTLY CUPY_PROCESSOR

camera = bettercam.create(output_idx=0, output_color="BGRA", torch_cuda=True) TO USE DIRECTLY TORCH_CUDA_PROCESSOR

camera = bettercam.create(output_idx=0, output_color="BGRA") TO USE DIRECTLY NUMPY_PROCESSOR

