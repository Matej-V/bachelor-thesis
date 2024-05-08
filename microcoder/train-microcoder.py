'''
This script trains a model on the CAUSAL modeling task using the dataset.

The script loads the prepared dataset, loads the model, tokenizer, and trains the model on the dataset.

Author: Matej Vadovic
Year: 2024
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training 
import os
from trl import SFTTrainer
import argparse
from traintools.lib import DatasetSingleton
import importlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on CAUSAL modeling task')
    parser.add_argument('config_class', type=str, help='Name of the config class to use')
    args = parser.parse_args()

    # Load config class from traintools.lib
    config = getattr(importlib.import_module('traintools.lib'), args.config_class)()

    os.environ["WANDB_PROJECT"] = config.wandb_project

    # Load prepared dataset dictionary
    train_dataset = DatasetSingleton.get_train_dataset()
    eval_dataset = DatasetSingleton.get_validation_dataset()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Map dataset to model input
    train_dataset = train_dataset.map(lambda x: {"train_example": f"{x['prompt']}\n{x['code']}"})
    eval_dataset = eval_dataset.map(lambda x: {"train_example": f"{x['prompt']}\n{x['code']}"})

    # Load model
    if config.bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=config.bnb_config,
            device_map=config.device_map,
            token=True,
            use_cache=False
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            token=True,
            use_cache=False
        )
    
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=config.lora_config,
        dataset_text_field="train_example",
        tokenizer=tokenizer,
        args=config.train_args[0],
        packing=False,
    )

    # Run training
    trainer.train()

    # Save final model
    trainer.model.save_pretrained(os.path.join(config.output_path, "final_checkpoint"))