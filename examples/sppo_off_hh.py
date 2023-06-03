import json
import math
import os
import sys

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from ppo_hh import create_reward_fn
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype
from datasets import load_from_disk

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SPPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import random

RANDOM_SEED = 1000
MODEL_SIZE = "1B"
LOSS = "log" # "square" or "log", square for APA, and log for AWR
ADV_COEFF_SQ = 1
ADV_COEFF_LOG = 1
OUTPUT_DIR = "output"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED) 
default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1000,
        epochs=10000,
        total_steps=20000,
        batch_size=4,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AccelerateSQLOffTrainer",
        checkpoint_dir="checkpoints/ppo_hh",
        seed = RANDOM_SEED,
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1e-6)),
    method=SPPOConfig(
        name="SPPOConfig",
        num_rollouts=64,
        chunk_size=16,
        ppo_epochs=2,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=100,
        cliprange_value=100,
        vf_coef=1,
        scale_reward="running",
        loss_str=LOSS,
        adv_coeff_sq=ADV_COEFF_SQ,
        adv_coeff_log=ADV_COEFF_LOG,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=100,
        gen_kwargs=dict(
            max_new_tokens=128,
            do_sample=True,
        ),
    ),
)


config_name = MODEL_SIZE  
if config_name == "125M":
    default_config.train.batch_size = 4
    default_config.method.chunk_size = 16
    default_config.train.total_steps = 20000
    default_config.model.model_path = "Dahoas/pythia-125M-static-sft" 
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.num_rollouts = 128
elif config_name == "1B":
    default_config.train.batch_size = 2
    default_config.train.total_steps = 20000
    default_config.optimizer.kwargs["lr"] = 1e-6
    default_config.scheduler.kwargs["eta_min"] = 1e-6
    default_config.model.model_path = "Dahoas/pythia-1B-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.chunk_size = 4
elif config_name == "6B":
    default_config.train.batch_size = 1
    default_config.train.total_steps = 20000
    default_config.model.model_path = "Dahoas/pythia-6B-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.chunk_size = 1
    default_config.optimizer.kwargs["lr"] = 1e-6
    default_config.scheduler.kwargs["eta_min"] = 1e-6
elif config_name == "20B":
    default_config.train.seq_length = 512
    default_config.train.batch_size = 1
    default_config.train.total_steps = 8000
    default_config.optimizer.kwargs["lr"] = 1e-6
    default_config.scheduler.kwargs["eta_min"] = 1e-6
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh_20B"
    default_config.model.model_path = "EleutherAI/gpt-neox-20b"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.num_rollouts = 16
    default_config.method.chunk_size = 4
    default_config.method.ppo_epochs = 2


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


reward_fn = create_reward_fn()
def preprocess(sample):
    sample["prompt_output"] = [
        [sample["prompt"], sample["chosen"]],
        [sample["prompt"], sample["rejected"]],
    ]
    prompt_res = [
        sample["prompt"] + sample["chosen"],
        sample["prompt"] + sample["rejected"],
    ]
    reward = reward_fn(prompt_res, sample["prompt"], sample["rejected"])

    sample["reward"] = reward
    return sample

def main(hparams={}):
    output_dir = OUTPUT_DIR
    config = TRLConfig.update(default_config, hparams)
    config.train.checkpoint_dir = output_dir 
    config.train.logging_dir = output_dir
    config.train.tracker = "tensorboard"
    
    
    
    dataset = load_dataset("Dahoas/rm-static").map(preprocess)
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])

    rewards = sum(dataset["train"]["reward"], [])

    eval_prompts = [prompt_output[0][0] for prompt_output in dataset["test"]["prompt_output"]][:280] #280


    trlx.train(
        samples=prompts_outputs,
        rewards=rewards,
        config=config,
        eval_prompts=eval_prompts,
        metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
