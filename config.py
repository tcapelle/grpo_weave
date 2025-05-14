from dataclasses import dataclass, field
from typing import Optional
import trl

@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    dataset_name: str = "gsm8k_dataset_prepared"
    model_name: str = "Qwen/Qwen3-4B"
    wandb_project: str = "grpo_weave"
    wandb_entity: Optional[str] = "grpo-cuda"