import torch
import paddle
from paddleocr import PaddleOCR

try:
    gpu_available = paddle.device.is_compiled_with_cuda()
    print("GPU available:", gpu_available)
    print(f" CuDA : {torch.cuda.is_available()}")

    ocr = PaddleOCR(lang="en", use_gpu=True, show_log=False)
    print(torch.version.cuda)
    print(paddle.device.get_device())
except Exception as e:
    print("Error", e)
