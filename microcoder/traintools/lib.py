'''
Library for training and evaluation of models

This library contains functions and configurations for training and evaluation of models used in the project.

Author: Matej Vadovic
Year: 2024
'''
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch
import numpy as np
import evaluate
from codebleu import calc_codebleu
from datetime import datetime
import pandas as pd
from typing import List, Dict, Optional
import regex as re
import datasets
import os

DATASET_PATH = os.getenv("DATASET_PATH")

base_codellama_model_name = "codellama/CodeLlama-7b-Instruct-hf"     # CodeLlama 7B Instruct model on Hugging Face
base_starcoder_model_name = "bigcode/starcoderbase-1b"               # StarCoderBase model on Hugging Face


class DatasetSingleton:
    '''
    Singleton class for loading the dataset

    This class ensures that the dataset is loaded only once and the same instance is returned every time it is accessed.
    '''
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatasetSingleton, cls).__new__(cls)
            if os.path.exists(DATASET_PATH):
                cls._instance.dataset = datasets.load_from_disk(DATASET_PATH)
                
            else:
                cls._instance.dataset = datasets.load_dataset('xvadov01/cpp_emb_nl2pl')
            cls._instance.name = DATASET_PATH
        return cls._instance

    @classmethod
    def get_dataset(cls):
        # Ensure the singleton instance is created if this is the first call
        if cls._instance is None:
            cls()
        return cls._instance.dataset
    
    @classmethod
    def get_train_dataset(cls):
        '''
        Get the training dataset split
        '''
        return cls.get_dataset()["train"]
    
    @classmethod
    def get_validation_dataset(cls):
        '''
        Get the validation dataset split
        '''
        return cls.get_dataset()["validation"]
    
    @classmethod
    def get_test_dataset(cls):
        '''
        Get the test dataset split
        '''
        return cls.get_dataset()["test"]

    @classmethod
    def get_name(cls):
        '''
        Get the name of the dataset
        '''
        return cls._instance.name


def predict(query: str, model: torch.nn.Module, tokenizer: torch.nn.Module, max_new_tokens:int = 2048, temperature: float = 0.3, repetition_penalty: Optional[float] = None, top_k: Optional[int] = None, top_p: Optional[float] = None, skip_special_tokens: bool = False) -> str:
    '''
    Generate text response from the model given a query

    Args:
    - query: input query
    - model: model to generate text
    - tokenizer: tokenizer for the model
    - max_new_tokens: maximum number of tokens to generate
    - temperature: temperature for sampling
    - repetition_penalty: repetition penalty for sampling
    - top_k: top k value for sampling
    - top_p: top p value for sampling
    - skip_special_tokens: whether to skip special tokens, EOS is always skipped

    Returns:
        Generated text response
    '''
    inputs = tokenizer(query, return_tensors='pt').to('cuda:0')
    output = tokenizer.decode(model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0], skip_special_tokens=skip_special_tokens)

    output = output.replace(tokenizer.eos_token, '')
    return output


def permute(sample: List[int], np_rng, prefix_tok_id:int, middle_tok_id:int, suffix_tok_id:int, pad_tok_id:int, fim_rate:int =1, fim_spm_rate:int=0.5):
    '''
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).

    Adapted from: https://github.com/loubnabnl/santacoder-finetuning/blob/main/fim.py
        
    Args:
    - sample: list of tokens
    - np_rng: numpy random number generator
    - suffix_tok_id: token id for suffix
    - prefix_tok_id: token id for prefix
    - middle_tok_id: token id for middle
    - pad_tok_id: token id for padding
    - fim_rate: probability of performing FIM transformation
    - fim_spm_rate: probability of using SPM mode
    - truncate_or_pad: whether to truncate or pad the sample to the original length after FIM transformation

    Returns:
        List of tokens after FIM transformation
    '''
    if np_rng.binomial(1, fim_rate):
        # Determine the points for roughly equal division with some randomness
        variability = max(1, len(sample) // 10)  # 10% of the sample length for variability

        third1 = len(sample) // 3
        third2 = 2 * len(sample) // 3

        # Calculate boundaries with added random variability
        boundary1 = np_rng.randint(third1 - variability, third1 + variability)
        boundary2 = np_rng.randint(third2 - variability, third2 + variability)

        # Ensure boundaries are within the list bounds and boundary1 is less than boundary2
        boundary1 = max(0, min(boundary1, len(sample) - 1))
        boundary2 = max(boundary1 + 1, min(boundary2, len(sample)))

        prefix = sample[:boundary1]
        middle = sample[boundary1:boundary2]
        suffix = sample[boundary2:]

        if np_rng.binomial(1, fim_spm_rate):
            # SPM mode
             new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # No FIM transformation is applied
        new_sample = sample

    return new_sample


def evaluate_predictions(predictions: List[str], references: List[List[str]], model_name:str, dataset_name:str, task:str, instruct: bool=False, tokenizer=None) -> Dict[str, float]:
    '''
    Evaluate the predictions using BLEU, CodeBLEU, ChrF++, ROUGE, and METEOR metrics

    Args:
    - predictions: list of predicted code snippets
    - references: list of reference code snippets
    - model_name: name of the model
    - dataset_name: name of the dataset
    - task: task type
    - instruct: whether the model is in instruct mode

    Returns:
        Dictionary containing the evaluation scores

    '''
    # Load metrics
    chrf = evaluate.load("chrf")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # Calculate different metrics
    print(f"Evaluating metrics...")
    result_record = {}
    result_record['chrf'] = chrf.compute(predictions=predictions, references=references, word_order=2, char_order=6)['score']
    result_record['bleu'] = bleu.compute(predictions=predictions, references=references)['bleu']
    result_record['codebleu'] = calc_codebleu(references, predictions, lang="cpp", weights=(0.25, 0.25, 0.25, 0.25))
    result_record['rouge'] = rouge.compute(predictions=predictions, references=references)
    result_record['meteor'] = meteor.compute(predictions=predictions, references=references)['meteor']

    # Add model, time, and dataset information
    result_record['model'] = model_name
    result_record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_record['dataset'] = f'{dataset_name}-len{len(predictions)}'
    result_record['task'] = task
    result_record['instruct'] = instruct
    print(f"Metrics evaluated. Scores saved to Results.csv")

    # Write results to a file
    with open('Results.csv', 'a') as f:
        pd.DataFrame([result_record]).to_csv(f, header=f.tell()==0)
    print("Results saved to Results.csv")

    return result_record


def remove_comments_from_output(text: str) -> str:
        '''
        Removing comments of different types from the text
        '''

        text = re.sub(r'[\s]*\/\*[^\*]*(?:\*\/)+', '', text, flags=re.DOTALL | re.MULTILINE)
        text = re.sub(r'[\s]*\/\/[^\n]*', '', text, flags=re.DOTALL | re.MULTILINE) 
        text = re.sub(r'\/\*[\s\S]*?\*\/', '', text)

        return text


def build_instruct_prompt(docstring: str, solution: Optional[str]) -> Dict[str, str]:
    '''
    Build an instruct prompt for the model

    Args:
    - docstring: description of the code
    - solution: code of the function

    Returns:
        Dictionary with the key 'train_example' and the value of the instruct ready example
    '''
    # Format from here https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF
    B_INST, E_INST = "[INST]", "[/INST]"
    if solution is not None:
        text = f"{B_INST}Write code to solve the following coding problem that obeys the constraints:\n{docstring}{E_INST}\n{solution}"
    else:
        text = f"{B_INST}Write code to solve the following coding problem that obeys the constraints:\n{docstring}{E_INST}"
    return {'train_example': text}


# Mapping of devices
device_map = {'': 0}

class BaseConfig:
    '''
    Base class for configurations
    '''
    def __init__(
        self,
        bnb_config = None,
        device_map = device_map,
        instruct = False,
        lora_config = None,
        max_new_tokens = None,
        model_name = None,
        n_samples = None,
        output_path = None,
        repetition_penalty = None,
        skip_special_tokens = True,
        temperature = None,
        tokenizer_name = None,
        top_k = None,
        top_p = None,
        torch_dtype = torch.float32,
        train_args = None,
        wandb_project = None
    ):
        self.bnb_config = bnb_config
        self.device_map = device_map
        self.instruct = instruct
        self.lora_config = lora_config
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.n_samples = n_samples
        self.output_path = output_path
        self.repetition_penalty = repetition_penalty
        self.skip_special_tokens = skip_special_tokens
        self.temperature = temperature
        self.tokenizer_name = tokenizer_name
        self.top_k = top_k
        self.top_p = top_p
        self.torch_dtype = torch_dtype
        self.train_args = train_args,
        self.wandb_project = wandb_project


class MicroCoderFIMTrain(BaseConfig):
    '''
    Configuration for training the MicroCoderFIM model
    '''
    def __init__(self):
        output_path = f"{base_starcoder_model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        super().__init__(
            model_name = base_starcoder_model_name,
            output_path = output_path,
            tokenizer_name = base_starcoder_model_name,
            train_args = TrainingArguments(
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=8,
                warmup_ratio=0.05,
                num_train_epochs=5,
                learning_rate=3e-5,
                lr_scheduler_type="cosine",
                logging_steps=500,
                optim="paged_adamw_32bit",
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=500,
                save_steps=2000,
                output_dir=output_path,
                load_best_model_at_end=False,
                group_by_length=True,
                report_to="wandb",
                run_name=output_path
            ),
            wandb_project = "starcoder"
        )


class MicroCoderTrain(BaseConfig):
    '''
    Configuration for training the MicroCoder model
    '''
    def __init__(self):
        output_path = f"{base_codellama_model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        super().__init__(
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4'
            ),
            instruct = False,    
            lora_config = LoraConfig(
                r=32,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            model_name = base_codellama_model_name,
            tokenizer_name = base_codellama_model_name,
            output_path = output_path,
            train_args = TrainingArguments(
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=1,
                warmup_ratio=0.05,
                bf16=True,
                num_train_epochs=3,
                learning_rate=1e-4,
                lr_scheduler_type="constant",
                logging_steps=100,
                optim="paged_adamw_32bit",
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=3000,
                output_dir=output_path,
                load_best_model_at_end=False,
                group_by_length=True,
                report_to="wandb",
                run_name=output_path,
            ),
            wandb_project = "codellama"
        )


class MicroCoderEval(BaseConfig):
    '''
    Configuration for evaluating the MicroCoder model
    '''
    def __init__(self):
        super().__init__(
            model_name = 'xvadov01/microcoder-7B-q4',
            tokenizer_name = 'xvadov01/microcoder-7B-q4',
            max_new_tokens = 2048,
            temperature = 0.3,
            n_samples = 1000,
            repetition_penalty = 1.3,
            skip_special_tokens = True
        )


class DeepSeekCoderEval(BaseConfig):
    '''
    Configuration for training the DeepSeekCoder model
    '''
    def __init__(self):
        super().__init__(
            model_name = 'deepseek-ai/deepseek-coder-6.7b-base',
            tokenizer_name = 'deepseek-ai/deepseek-coder-6.7b-base',
            torch_dtype = torch.float16,
            max_new_tokens = 300,
            temperature = 0.3
        )


class MicroCoder16bEval(BaseConfig):
    '''
    Configuration for evaluating the MicroCoder model
    '''
    def __init__(self):
        super().__init__(
            model_name = 'xvadov01/microcoder-16b',
            max_new_tokens = 512,
            n_samples = 1000,
            repetition_penalty = 1.3,
            skip_special_tokens = True,
            temperature = 0.3,
            tokenizer_name = 'xvadov01/microcoder-16b',
            torch_dtype = torch.float16
        )


class MicroCoderFIMEval(BaseConfig):
    '''
    Configuration for evaluating the MicroCoder model
    '''
    def __init__(self):
        super().__init__(
            model_name = 'xvadov01/microcoderfim-1B',
            max_new_tokens = 128,
            skip_special_tokens = False,
            temperature = 0.2,
            tokenizer_name='xvadov01/microcoderfim-1B'
        )


class Test100MCFIMConfig(BaseConfig):
    '''
    Configuration for evaluating the MicroCoder model
    '''
    def __init__(self, max_new_tokens, temperature, top_p):
        super().__init__(
            model_name = 'xvadov01/microcoderfim-1B',
            max_new_tokens = max_new_tokens,
            n_samples = 250,
            skip_special_tokens = False,
            temperature = temperature,
            top_p = top_p,
        )