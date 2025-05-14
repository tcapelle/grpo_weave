import trl
import torch
import wandb
import weave
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, TrlParser, GRPOTrainer

import simple_parsing as sp

from config import GRPOScriptArguments
from rewards import reward_correct, reward_format

def main(script_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # create an explicit accelerator
    accelerator = Accelerator()

    # Load the dataset
    dataset = load_from_disk(script_args.dataset_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    # pass our reward functions
    reward_funcs = [reward_correct, reward_format]

    if accelerator.is_main_process:
        wandb.init(
            project=script_args.wandb_project, 
            entity=script_args.wandb_entity)

        weave.init(script_args.wandb_project)

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()

    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["grpo_weave"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


    # push to hub
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)