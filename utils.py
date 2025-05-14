import wandb
import weave
from contextlib import nullcontext

def wandb_attributes():
    "Add the wandb metrics as weave attributes"
    if wandb.run is None:
        return nullcontext()
    else:
        run = wandb.run
        wandb_metrics = {k: v for k, v in dict(run.summary).items() if not k.startswith("_")}
        return weave.attributes(wandb_metrics)