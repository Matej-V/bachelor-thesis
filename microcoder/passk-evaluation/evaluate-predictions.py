'''
This script evaluates the predictions made by the model on the test set of the OpenAI Humaneval dataset. It uses the code_eval library to compute the evaluation metrics. The predictions are loaded from a JSON file, and the references are taken from the test set of the dataset.

Use this code in safe environments only and set HF_ALLOW_CODE_EVAL=1 in environment variables to enable code evaluation.

Author: Matej Vadovic
Year: 2024
'''
import datasets
import evaluate
import json
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    code_eval = evaluate.load('code_eval')
    dataset = datasets.load_dataset("openai_humaneval")['test']
    predictions = json.load(open('fim-outputs.json'))
    entry_points = dataset['entry_point']
    references = dataset['test']
    test_cases = [reference + f'check({entry_point})' for reference, entry_point in zip(references, entry_points)]

    results = code_eval.compute(predictions=predictions, references=test_cases)

    # Save results to a file
    with open('fim-results.json', 'w') as f:
        json.dump(results[0], f)

    for k,v in results[0].items():
        print(f"{k}: {v}")

    