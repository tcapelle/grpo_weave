from dataclasses import dataclass, field
from typing import Optional
import trl

@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    dataset_name: str = "gsm8k_dataset_prepared"
    model_name: str = "Qwen/Qwen3-4B"
    wandb_project: str = "simple_grpo"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None