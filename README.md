Quantized Model for Runnning On FPGA
=================
This is a team project repository of the FPGA Convolution accelerator implementation.
This contains information about models running on the device.

Directory structure
------------------
Directory structure is shown belows :

```
├── models                                   # Model files
│   ├── caffe                                # TF2Caffe converted models for the simulator
│   └── tensorflow                           # Tensorflow model files
│       ├── efficientnet-lite0_int8.tflite   # Modified tensorflow models for better accuracy
│       └── official                         # Official model files
├── caffe                                    # Modified caffe to verify the results of FPGA
│                                            # (=Fixed-point simulator)
└── scripts(To be updated)                   # Python scripts files
```


Base Model: [EfficientNet-lite](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
------------------
We chose "EfficientNet-lite" because it is a mobile/IoT friendly image classification models. Notably, especially, the lite version of EfficientNet is optimized for all mobile CPU/GPU/EdgeTPU.

Significant features of EfficientNet-lite are as follows.

* Remove squeeze-and-excite (SE): SE are not well supported for some mobile accelerators.
* Replace all swish with RELU6: for easier post-quantization.
* Fix the stem and head while scaling models up: for keeping models small and fast.

Here is the official checkpoints of Efficient-lite0 what we used, and its accurracy, params, flops, and Pixel4's CPU/GPU latency.

|**Model** | **params** | **MAdds** | **FP32 accuracy (official)** | **FP32 accuracy (measured)** | **FP32 CPU  latency** | **FP32 GPU latency** | **INT8 accuracy (official)** |**INT8 accuracy (measured)** | **INT8 CPU latency**  |
|------|-----|-------|-------|-------|-------|-------|-------|-------|-------|
|efficientnet-lite0 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz) | 4.7M | 407M |  75.1% | 74.91% | 12ms | 9.0ms | 74.4% | 74.78% |  6.5ms |


Fixed-point Simulator
------------------
To verify the operation on the FPGA, the golden vectors accurate in bit-level is needed.
Since the default tensorflow's quantization scheme uses quantized fp32 values, bit-level differences can be caused from the results of the FPGA.
So we implemented a framework-level fixed-point simulator with [NVCaffe](https://github.com/NVIDIA/caffe).

Most of CUDA codes are not imeplemented, so you'd better to use CPU.

