'''
Script used to run and evaluate model on FIM task

The script for 

Author: Matej Vadovic
Year: 2024
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re
import argparse
from traintools.lib import predict, permute, DatasetSingleton, evaluate_predictions
import importlib

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate MicroCoder7B model')
    parser = argparse.ArgumentParser(description='Train model on CAUSAL modeling task')
    parser.add_argument('config_class', type=str, help='Name of the config class to use')
    args = parser.parse_args()

    # Load config class from traintools.lib
    config = getattr(importlib.import_module('traintools.lib'), args.config_class)()

    # Load dataset dict
    test_dataset =  DatasetSingleton.get_test_dataset()
    if config.n_samples is not None:
        print(f"Using {config.n_samples} samples")
        test_dataset = test_dataset[:config.n_samples]

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        trust_remote_code=True,
        token=True,
        return_dict=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Get special tokens for FIM from the tokenizer
    FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.all_special_ids[1:5]

    # Create numpy random number generator
    np_rng = np.random.RandomState()    

    # Create permuted inputs for the test dataset
    inputs = [permute(
        tokenizer(f'{prompt}\n{code}', return_tensors="np")['input_ids'][0],
        np_rng,
        FIM_PREFIX,
        FIM_MIDDLE,
        FIM_SUFFIX,
        FIM_PAD,
        1,
        0.5
    ) for prompt, code in zip(test_dataset['prompt'], test_dataset['code'])]

    # Decode permuted inputs
    inputs = [tokenizer.decode(ex) for ex in inputs]

    # Reference middle parts that are from the original dataset
    references = [[input[re.search(r'<fim_middle>', input).end():]] for input in inputs]

    # Prompt is everything up to the FIM_MIDDLE token
    prompts = [ex[:re.search(r'<fim_middle>', ex).end()] for ex in inputs]

    print('Generating predictions...')
    # Generate predictions, so predicting middle parts, since prefix and suffix are given before middle (SPM, PSM)
    predictions = [
        (lambda x: x[re.search(r'<fim_middle>', x).end():])(predict(
            prompt,
            model,
            tokenizer,
            max_new_tokens=config.max_new_tokens,
            top_p=config.top_p,
            temperature=config.temperature,
        ))
        for prompt in prompts]
    print('Predictions generated')

    # Evaluate predictions
    result_record = evaluate_predictions(predictions, references, config.model_name, DatasetSingleton.get_name(), 'FIM', tokenizer=tokenizer)
    print(result_record)