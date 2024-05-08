'''
Fine-tuning script for StarCoder1B model on differentiable programming task

Full finetuning of model on differentiable programming task using StarCoder1B model. The script loads the prepared dataset dictionary, loads the model, tokenizer, and training arguments, and trains the model on the dataset. The script also supports fine-tuning on FIM by adding FIM examples to the dataset.

Author: Matej Vadovic
Year: 2024
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
from trl import SFTTrainer
from traintools.lib import permute, DatasetSingleton
import argparse
from typing import Any, Dict
import importlib


def prep_fim_example(docstring: str, code: str, np_rng: np.random.Generator, tokenizer: Any) -> Dict[str, str]:
    '''
    Combines docstring and code into a single training example for FIM fine-Tuning

    Args:
    - docstring: description of the code
    - code: code of the function
    - np_rng: random number generator

    Returns:
        Dictionary with the key 'train_example' and the value of the FIM ready example
    '''
    FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.all_special_ids[1:5]
    # Add description to code and add FIM tokens
    example = permute(
        tokenizer(f'{docstring}\n{code}', return_tensors="np")['input_ids'][0],
        np_rng,
        FIM_PREFIX,
        FIM_MIDDLE,
        FIM_SUFFIX,
        FIM_PAD,
    )
    return {'train_example': tokenizer.decode(example)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train starcoder model on FIM programming task")
    parser.add_argument('config_class', type=str, help='Name of the config class to use')
    arg = parser.parse_args()

    # Load config class from traintools.lib
    config = getattr(importlib.import_module('traintools.lib'), arg.config_class)()

    os.environ["WANDB_PROJECT"] = config.wandb_project

    # Load prepared dataset dictionary
    train_dataset = DatasetSingleton.get_train_dataset()
    eval_dataset = DatasetSingleton.get_validation_dataset()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    #  Prepare FIM examples
    np_rng = np.random.RandomState()

    # Add FIM examples to the dataset
    train_dataset = train_dataset.map(lambda x: prep_fim_example(x["prompt"], x["code"], np_rng, tokenizer))
    eval_dataset = eval_dataset.map(lambda x: prep_fim_example(x["prompt"], x["code"], np_rng, tokenizer))
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        trust_remote_code=True,
        token=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="train_example",
        tokenizer=tokenizer,
        args=config.train_args[0],
        packing=False,
    )

    # Run training
    trainer.train()

    # Save final model
    trainer.model.save_pretrained(os.path.join(config.output_path, "final_checkpoint"))