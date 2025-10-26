python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen3 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.96 \
    --max-model-len 45000
