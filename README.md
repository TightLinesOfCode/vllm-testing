# vLLM Testing

## Serving Nemotron 120B NVFP4

```bash
docker run -d --gpus all --restart unless-stopped -v ~/.cache/huggingface:/root/.cache/huggingface --shm-size=16g -p 8989:8000 -e HUGGING_FACE_HUB_TOKEN=[token] -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm/vllm-openai:v0.17.1-cu130 --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --dtype auto --kv-cache-dtype fp8 --trust-remote-code --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --enable-expert-parallel
```

### Docker Parameters

| Parameter | Description |
|---|---|
| '-d' | Runs the docker container in detached mode. ( Background process } |
| `--gpus all` | Exposes all NVIDIA GPUs to the container via the NVIDIA Container Toolkit. |
| `--restart unless-stopped` | Automatically restarts the container if it crashes, unless you explicitly stop it. Survives host reboots. |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | Mounts the host's Hugging Face cache into the container so model weights are downloaded once and reused across runs. |
| `--shm-size=16g` | Sets shared memory to 16 GB. Required for PyTorch's NCCL multi-GPU communication which uses `/dev/shm` for inter-process data transfer. |
| `-p 8989:8000` | Maps host port 8989 to the container's port 8000, where the vLLM OpenAI-compatible API server listens. |
| `-e HUGGING_FACE_HUB_TOKEN=[token]` | Provides your Hugging Face access token for downloading gated models. |
| `-e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` | Selects the FlashInfer-CUTLASS backend for NVFP4 matrix multiplications, which is optimized for Blackwell GPUs. |
| `-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | Allows the model to use its full context length even when vLLM would otherwise cap it due to memory constraints. |

### vLLM Parameters

| Parameter | Description |
|---|---|
| `vllm/vllm-openai:v0.17.1-cu130` | The vLLM Docker image with CUDA 13.0 support (required for Blackwell architecture GPUs). |
| `--model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | The Hugging Face model to serve. This is NVIDIA's 120B parameter Mixture-of-Experts model quantized to NVFP4 (4-bit floating point), with 12B active parameters per forward pass. |
| `--dtype auto` | Automatically selects the appropriate data type for model weights based on the model config. |
| `--kv-cache-dtype fp8` | Stores the key-value cache in FP8 format, cutting KV cache memory usage in half compared to FP16 while preserving quality. |
| `--trust-remote-code` | Allows execution of custom modeling code from the Hugging Face repo. Required for models with non-standard architectures. |
| `--gpu-memory-utilization 0.9` | Allocates up to 90% of each GPU's VRAM for the model and KV cache, reserving 10% as headroom. |
| `--tensor-parallel-size 2` | Shards the model across 2 GPUs using tensor parallelism, splitting individual layers across devices. |
| `--enable-expert-parallel` | Distributes MoE (Mixture-of-Experts) expert layers across GPUs, so each GPU handles a subset of experts rather than duplicating them all. |
