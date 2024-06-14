import time
import bettercam

def benchmark_nvidia_gpu():
    TOP = 0
    LEFT = 0
    RIGHT = 1920
    BOTTOM = 1080
    region = (LEFT, TOP, RIGHT, BOTTOM)
    title = "[BetterCam] Capture benchmark with NVIDIA GPU"

    camera = bettercam.create(output_idx=0, output_color="BGRA", nvidia_gpu=True)
    camera.start(target_fps=0, video_mode=True)

    start_time = time.time()

    for i in range(1000):
        image = camera.get_latest_frame()

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1000 / elapsed_time

    camera.stop()
    del camera

    print(f"{title}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Frames per second: {fps:.2f} FPS")
    return elapsed_time, fps

def benchmark_torch_cuda():
    TOP = 0
    LEFT = 0
    RIGHT = 1920
    BOTTOM = 1080
    region = (LEFT, TOP, RIGHT, BOTTOM)
    title = "[BetterCam] Capture benchmark with Torch CUDA"

    camera = bettercam.create(output_idx=0, output_color="BGRA", torch_cuda=True)
    camera.start(target_fps=0, video_mode=True)

    start_time = time.time()

    for i in range(1000):
        image = camera.get_latest_frame()

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1000 / elapsed_time

    camera.stop()
    del camera

    print(f"{title}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Frames per second: {fps:.2f} FPS")
    return elapsed_time, fps

def benchmark_no_gpu():
    TOP = 0
    LEFT = 0
    RIGHT = 1920
    BOTTOM = 1080
    region = (LEFT, TOP, RIGHT, BOTTOM)
    title = "[BetterCam] Capture benchmark without GPU acceleration"

    camera = bettercam.create(output_idx=0, output_color="BGRA")
    camera.start(target_fps=0, video_mode=True)

    start_time = time.time()

    for i in range(1000):
        image = camera.get_latest_frame()

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1000 / elapsed_time

    camera.stop()
    del camera

    print(f"{title}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Frames per second: {fps:.2f} FPS")
    return elapsed_time, fps

# Benchmark with NVIDIA GPU
nvidia_time, nvidia_fps = benchmark_nvidia_gpu()

# Benchmark with Torch CUDA
torch_time, torch_fps = benchmark_torch_cuda()

# Benchmark without GPU acceleration
no_gpu_time, no_gpu_fps = benchmark_no_gpu()

# Print comparison results
print("\nComparison Results:")
print(f"NVIDIA GPU - Elapsed time: {nvidia_time:.2f} seconds, FPS: {nvidia_fps:.2f}")
print(f"Torch CUDA - Elapsed time: {torch_time:.2f} seconds, FPS: {torch_fps:.2f}")
print(f"No GPU - Elapsed time: {no_gpu_time:.2f} seconds, FPS: {no_gpu_fps:.2f}")


