from dataclasses import dataclass, field

import trl

@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    dataset_name: str = "gsm8k_dataset_prepared"
    model_name: str = "Qwen/Qwen3-4B"