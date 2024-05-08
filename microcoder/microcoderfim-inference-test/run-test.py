'''
This script runs predictions on the FIM test dataset and evaluates them. It uses the Test100MCFIMConfig configuration class from traintools.lib. It runs predictions for different values of max_new_tokens, top_p and temperature. The results are saved in temp-records.json and top-p-records.json files.

Author: Matej Vadovic
Year: 2024
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re
import json
from traintools.lib import predict, permute, DatasetSingleton, evaluate_predictions
import importlib


def run_prediction(config_class, max_new_tokens, top_p, temperature):
    # Load config class from traintools.lib
    config = getattr(importlib.import_module('traintools.lib'), config_class)(max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature)

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
    return evaluate_predictions(predictions, references, config.model_name, DatasetSingleton.get_name(), 'FIMTest', tokenizer=tokenizer)


# Run predictions for different values of max_new_tokens, top_p and temperature
if __name__ == '__main__':
    temperatures = [0.1, 0.2, 0.5, 0.8]
    top_ps = [0.5, 0.9]
    max_new_tokens_values = [128, 256, 512]

    temp_records = {
        str(temperature): {
            str(max_new_tokens): run_prediction('Test100MCFIMConfig', max_new_tokens, None, temperature)
            for max_new_tokens in max_new_tokens_values
        }
        for temperature in temperatures
    }

    top_p_records = {
        str(top_p): {
            str(max_new_tokens): run_prediction('Test100MCFIMConfig', max_new_tokens, top_p, None)
            for max_new_tokens in max_new_tokens_values
        }
        for top_p in top_ps
    }

    with open('temp-records.json', 'w') as f:
        json.dump(temp_records, f)

    with open('top-p-records.json', 'w') as f:
        json.dump(top_p_records, f)


