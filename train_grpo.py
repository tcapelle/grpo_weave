from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import re
import wandb
from math_verify import parse, verify, ExprExtractionConfig
from datasets import load_dataset, Dataset


model = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model)



training_args = GRPOConfig(
    use_vllm = True,
    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "cuda:0",
    },
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    beta=0.0,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    log_completions = True,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "grpo_trl_output",
)

wandb.init(project="grpo-trl", config=training_args)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [reward_correct, reward_format],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()