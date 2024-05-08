'''

Author: Matej Vadovic
Year: 2024
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from traintools.lib import evaluate_predictions, predict, build_instruct_prompt, remove_comments_from_output, DatasetSingleton
import importlib

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate MicroCoder7B model')
    parser.add_argument('config_class', type=str, help='Name of the config class to use')
    args = parser.parse_args()

    # Load config class from traintools.lib
    config = getattr(importlib.import_module('traintools.lib'), args.config_class)()

    # Load prepared dataset dictionary
    test_dataset = DatasetSingleton.get_test_dataset()

    # Load the model
    if config.bnb_config is not None:
        print("Using quantization")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            quantization_config=config.bnb_config,
            trust_remote_code=True,
            token=True,
            return_dict=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
            token=True,
            return_dict=True
        )

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    B_INST, E_INST = "[INST]", "[/INST]"

    if config.instruct:
        print("Using instruct mode")
        test_dataset = test_dataset.map(lambda x: build_instruct_prompt(x['prompt']))
    else:
        print("Using non-instruct mode")
        test_dataset = test_dataset.map(lambda x: {"train_example": f"{x['prompt']}\n"})

    if config.n_samples is not None:
        print(f"Using {config.n_samples} samples")
        test_dataset = test_dataset[:config.n_samples]

    # Generate predictions
    print(f"Generating predictions...")
    predictions = [
        predict(
            input,
            model,
            tokenizer,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=config.repetition_penalty,
            skip_special_tokens=config.skip_special_tokens,
        ).replace(input, '') for input in test_dataset['train_example']
    ]
    print("Predictions generated")

    if config.instruct:
        # Remove comments if using instruct mode
        predictions = [remove_comments_from_output(x) for x in predictions]

    # Reference predictions
    references = [[x] for x in test_dataset['code']]

    # Evaluate predictions
    result_record = evaluate_predictions(predictions, references, config.model_name, DatasetSingleton.get_name(), 'CAUSAL_LM', config.instruct, tokenizer=tokenizer)
    print(result_record)