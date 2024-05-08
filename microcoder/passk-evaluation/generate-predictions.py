from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from ..traintools.lib import predict, base_starcoder_model_name
import json

# Load prepared dataset dictionary
test_dataset = load_dataset('openai_humaneval')['test']

prompts = test_dataset['prompt']
references = [[x] for x in test_dataset['canonical_solution']]

tokenizer_sc = AutoTokenizer.from_pretrained(base_starcoder_model_name, padding_side='left')
tokenizer_sc.pad_token = tokenizer_sc.eos_token

model_sc = AutoModelForCausalLM.from_pretrained(
    'microcoderfim',
    device_map={'': 0},
    token=True,
    return_dict=True,
)

print('Generating outputs...')
outputs = [
    [predict(
        '<fim_prefix>' + prompt + '<fim_suffix>' + '<fim_middle>',
        model_sc,
        tokenizer_sc,
        max_new_tokens=128,
        temperature=0.2,
        skip_special_tokens=True,
    ) for i in range(200)] for prompt in prompts
]

# Dump outputs to json
with open('fim-outputs.json', 'w') as f:
    json.dump(outputs, f)
