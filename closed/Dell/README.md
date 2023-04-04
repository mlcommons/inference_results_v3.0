# MLPerf Inference v3.0 Implementations
This is a repository of Dell Technologies servers using optimized implementations for [MLPerf Inference Benchmark v2.1](https://www.mlperf.org/inference-overview/).

# Implementations
## Benchmarks
**Please refer to /closed/NVIDIA for detailed instructions for NVIDIA GPU & Triton submissions, including performace guides, and instructions on how to run with new systems.** 

**Please refer to /closed/Qualcomm for detailed instructions for Qualcomm Cloud AI 100 submissions.**

**Please refer to /closed/Intel for detailed instructions for Intel CPU submissions.**
  
The following benchmarks are part of our submission for MLPerf Inference v3.0:
- [3d-unet](code/3d-unet/tensorrt/README.md)
- [bert](code/bert/tensorrt/README.md)
- [dlrm](code/dlrm/tensorrt/README.md)
- [rnnt](code/rnnt/tensorrt/README.md)
- [retinanet](code/retinanet/README.md)
- [resnet50](code/resnet50/tensorrt/README.md)

# Dell Technologies Submission Systems

The closed systems that Dell has submitted on are:
- Datacenter Systems
  - Dell PowerEdge R750xa
    - NVIDIA A100-PCIe-80GB
    - NVIDIA H100-PCIe-80GB
    - NVIDIA H100-80C (virtualized)
  - Dell PowerEdge R760
    - Intel(R) Xeon(R) Platinum 8480+
  - Dell PowerEdge XE2420
    - NVIDIA T4
  - Dell PowerEdge XE8545
    - NVIDIA A100-SXM-80GB / 500W
    - NVIDIA A100-SXM-80C (virtualized)
  - Dell PowerEdge XE9680
    - NVIDIA A100-SXM-80GB / 500W
    - NVIDIA H100-SXM-80GB
  - Dell PowerEdge XR4520c
    - NVIDIA A30
- Edge Systems
  - Dell PowerEdge R650
    - Qualcomm Cloud AI Standard
  - Dell PowerEdge XE2420
    - NVIDIA T4
  - Dell PowerEdge XR4520c
    - NVIDIA A2
    - NVIDIA A2 MaxQ
    - NVIDIA A30
    - Qualcomm Cloud AI Lite
  - Dell PowerEdge XR5610
    - NVIDIA L4
  - Dell PowerEdge XR7620
    - NVIDIA L4

