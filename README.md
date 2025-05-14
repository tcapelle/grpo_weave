# grpo_weave


## vLLM server

Use something like tmux if you want to run on a single machine to ensure the terminals are kept alive.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --tensor-parallel-size 4 --model Qwen/Qwen3-4B
```

```
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train.py --config simple_config.yaml
```