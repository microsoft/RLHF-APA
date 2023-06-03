import json
import math
import os
import sys

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn

from reward_model.reward_model import GPTRewardModel

from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

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

OUTPUT_DIR = "output"

RANDOM_SEED = 1000
MODEL_SIZE = "6B"
LOSS = "square" # "square" or "log", square for APA, and log for AWR
ADV_COEFF_SQ = 10
ADV_COEFF_LOG = 1

REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/pytorch_model.bin"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
    )
SFT_MODEL_PATH = "CarperAI/openai_summarize_tldr_sft"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED) 
default_config = TRLConfig(
    train=TrainConfig(
        seq_length=550,
        epochs=10000,
        total_steps=5000,
        batch_size=4,
        checkpoint_interval=1000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateSPPOTrainer",
        checkpoint_dir="checkpoints/ppo_tldr",
        seed=RANDOM_SEED,
    ),
    model=ModelConfig(model_path="CarperAI/openai_summarize_tldr_sft", num_layers_unfrozen=8),
    tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1e-6)),
    method=SPPOConfig(
        name="SPPOConfig",
        num_rollouts=128,
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
        ref_mean=None,
        ref_std=None,
        loss_str=LOSS,
        adv_coeff_sq=ADV_COEFF_SQ,
        adv_coeff_log=ADV_COEFF_LOG,        
        cliprange_reward=100,
        gen_kwargs=dict(
            max_new_tokens=50,
        ),
    ),
)
 


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def create_reward_fn(post_summary_dict, config):  # noqa:  C901
    reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="/mnt/data/clausa-rl/tokenizers/gpt-j-6B")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    triton_host = os.environ.get("TRITON_HOST")

    if triton_host:
        triton_url, triton_model = triton_host.split("/")
        client = client_util.InferenceServerClient(url=triton_url, verbose=False)

        def reward_fn(samples, prompts, outputs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            input = reward_tokenizer(samples, padding=True, max_length=1024)

            mbs = 24
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)

                result = client.infer(triton_model, [prepare_tensor("input_ids", input_ids)])
                rewards = result.as_numpy("rewards")
                out.extend(rewards)

            return out

    elif os.environ.get("RANK", "0") == "0":
        reward_model = GPTRewardModel(SFT_MODEL_PATH, cache_dir="/mnt/data/clausa-rl/models/sft_summarization_gptj_tldr")
        reward_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
        reward_model.eval()
        reward_model.requires_grad_(False)
        device = torch.cuda.device_count() - 1
        reward_model = reward_model.half().to(device)

        def get_scores(samples):
            scores_list = []
            batch_size = 1
            for i in range(0, len(samples), batch_size):
                sub_samples = samples[i : i + batch_size]
                sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
                encodings_dict = reward_tokenizer(
                    sub_samples,
                    truncation=True,
                    max_length=config.train.seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encodings_dict["input_ids"].to(device)
                attn_masks = encodings_dict["attention_mask"].to(device)
                input_ids = input_ids.repeat(2, 1)
                attn_masks = attn_masks.repeat(2, 1)
                with torch.no_grad():
                    sub_scores = reward_model(input_ids=input_ids, attention_mask=attn_masks)
                scores_list.append(sub_scores["chosen_end_scores"])
            scores = torch.cat(scores_list, dim=0)
            return scores

        def reward_fn(samples, **kwargs):
            # original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
            # for text in original_samples:
            #     try:
            #         original_samples = [text + post_summary_dict[text.strip()]]
            #     except:
            #         print("ERROR: \n {0}".format(text.strip()))

            # original_scores = get_scores(original_samples)
            scores = get_scores(samples)
            norms_scores = scores # - original_scores
            return norms_scores

    else:
        reward_fn = True

    return reward_fn


def main(hparams={}):

    default_config.train.batch_size = 2
    default_config.train.total_steps = 5000
    default_config.model.model_path = SFT_MODEL_PATH 
    default_config.method.chunk_size = 1

    output_dir = OUTPUT_DIR
    config = TRLConfig.update(default_config, hparams)
    # config.train.rollout_logging_dir = output_dir
    config.train.checkpoint_dir = output_dir
    config.train.logging_dir = output_dir
    config.train.tracker = "tensorboard"

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    def get_prompt_dataset(tokenizer, prompts, max_length):
        formatted_prompts = []
        for i in range(len(prompts)):
            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i].split("TL;DR:")[0],
                    truncation=True,
                    max_length=max_length - 5,  # to make sure "TL;DR" dont get truncated
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = tmp + "\nTL;DR:"
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    dataset = load_dataset("CarperAI/openai_summarize_tldr")

    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]

    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    post_summary_dict = {}
    train_prompts = get_prompt_dataset(tokenizer, train_posts, max_length_input)

    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]
    val_prompts = get_prompt_dataset(tokenizer, val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]

    reward_fn = create_reward_fn(post_summary_dict, config)

    trlx.train(
        prompts=train_prompts,
        eval_prompts=val_prompts[0:500],
        reward_fn=reward_fn,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
