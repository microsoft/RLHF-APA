import json
import os
import uuid
from time import time
from typing import Callable, List

import ray
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_sppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.offline_pipeline import ( 
    tokenize_dialogue,
)
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import Clock, infinite_dataloader
from trlx.utils.modeling import RunningMoments, logprobs_of_labels
from copy import deepcopy

logger = logging.get_logger(__name__)


@register_trainer
class AccelerateSQLOffTrainer(AccelerateRLTrainer):
    """PPO Accelerate Trainer"""

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """PPO Accelerate Trainer initialization

        Args:
            config: Config
        """
        super().__init__(config, **kwargs)

        self.log_rollouts = False

        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        self.store = PPORolloutStorage(self.tokenizer.pad_token_id)

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(self.config.train.batch_size, shuffle=True)
        device = torch.cuda.device_count() - 3
        self.ref_model = deepcopy(self.model).eval().to(device) 
        # Prepare multi-GPU acceleration
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Setup a reference model when hydra heads are not used
        # if not hasattr(self.model, "frozen_head"):
        #     self.ref_model = self.get_arch(self.config)
        #     self.ref_model.to(self.accelerator.device)
        #     self.ref_model.eval()

        
        # Setup the KL controller
        # This helps prevent large divergences in the controller (policy)
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(config.method.init_kl_coef, config.method.target, config.method.horizon)
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        if config.model.model_arch_type == "seq2seq":
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.generate_experience_kwargs = None
        else:
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                self.generate_experience_kwargs = None

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config

        return from_fn(
            config.model.model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )

    def loss(self, batch: PPORLBatch):
        """Forward pass & loss

        Args:
            batch: Previous batch of episodes
        """
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)
        response_length = old_rewards.shape[1]

        advantages, returns = self.config.method.get_advantages_and_returns(old_values, old_rewards, response_length)

        if self.config.model.model_arch_type == "seq2seq":
            input_ids = query_tensors
            decoder_input_ids = response_tensors
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            decoder_attention_mask = (
                decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            )
            decoder_attention_mask[:, 0] = 1

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            logits = outputs.logits
            values_pred = outputs.value
            logprobs = logprobs_of_labels(logits[:, :-1, :], decoder_input_ids[:, 1:])
            mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            start = 0
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                mask[:, start:end],
            )
        else:
            tokens = torch.cat((query_tensors, response_tensors), dim=1)
            attention_mask = tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
            outputs = self.model(tokens, attention_mask, return_dict=True)
            logits = outputs.logits            
            logsumexp = torch.logsumexp(logits, dim=-1)
            values_pred = outputs.value
            values_pred = values_pred[:, :-1]
            logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

            start = query_tensors.shape[1] - 1
            end = start + response_length
            logprobs, values_pred, mask, logsumexp = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start:end],
                logsumexp[:, start:end],
            )

        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
            logsumexp=logsumexp,
        )
        
        return loss, stats

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(config.train.rollout_logging_dir, self.run_id)
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    # def post_epoch_callback(self):
    #     """Post epoch callback

    #     Clears the store and creates `num_rollouts` new episodes.
    #     """
         
        # if self.log_rollouts:
        #    self.store.export_history(location=self.rollout_logging_dir)
        # self.store.clear_history()
        # Collect more rollouts for training
        # Eself.make_experience(self.config.method.num_rollouts, self.iter_count)

    # def post_backward_callback(self):
    #     self.kl_ctl.update(self.mean_kl.item(), n_steps=self.config.train.batch_size)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)
        self.train_dataloader = self.store.create_loader(self.config.train.batch_size, shuffle=True)

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = self.config.train.epochs * self.n_updates_per_batch * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    # def add_prompt_pipeline(self, pipeline: PromptPipeline):
    #     """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
    #     prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=True)
    #     prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
    #     self.prompt_iterator = infinite_dataloader(prompt_dataloader)


    def make_experience(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """

        if self.config.model.model_arch_type == "seq2seq":
            return self.make_experience_seq2seq(samples, rewards, max_length)
        logger.info("Collecting rollouts")
        # print(samples, len(samples))
        # print(rewards, len(rewards))
        if self.tokenizer:
            samples = [tokenize_dialogue(s, self.tokenizer, max_length) for s in samples]
        # print("tokenized", samples, len(samples))
        ppo_rl_elements = []
        all_input_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        device = self.model.device
        
        for sample in samples:
            length = 0
            all_input_ids.append(torch.tensor(sum(sample, [])).to(device))
            isoutput = False
            actions_ixs = []
            # print("sample", sample, len(sample))
            for phrase in sample:
                # print("phrase", phrase)
                if isoutput:
                    actions_ixs.append(torch.arange(length - 1, length + len(phrase) - 1))
                # print("actions_ixs", actions_ixs)
                length += len(phrase)
                isoutput = not isoutput

            states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)
            
        attention_mask = [torch.ones(len(x), dtype=int).to(device) for x in all_input_ids]
        ref_device = torch.cuda.device_count() - 3
        sample_lengths = np.array(list(map(len, all_input_ids)))
        output_lengths = np.array(list(map(len, all_actions_ixs)))
        prompt_lengths = sample_lengths - output_lengths
        returns = torch.tensor(rewards, dtype=float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-30)
        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret
        with torch.no_grad():
            n_samples: int = len(samples)
            for sample_idx in range(n_samples):
                
                *_, values = self.model(
                            torch.unsqueeze(all_input_ids[sample_idx], 0),
                            attention_mask=torch.unsqueeze(attention_mask[sample_idx], 0),
                )
                ref_logits = self.ref_model(
                            torch.unsqueeze(all_input_ids[sample_idx], 0).to(ref_device),
                            attention_mask=torch.unsqueeze(attention_mask[sample_idx], 0).to(ref_device),
                            return_dict=True,
                ).logits
                # ref_logits = ref_logits

                # print("ref_logits", ref_logits, ref_logits.shape)
                # print("all_input_ids", all_input_ids[sample_idx], all_input_ids[sample_idx].shape)
                
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], torch.unsqueeze(all_input_ids[sample_idx][1:], 0).to(ref_device)).cpu()
 
                # Get the logprobs and values, for tokens that are not padding
                # or beginning of sequences tokens. These are from the model (not the reference model)
                # all_values = [values[ix, prompt_lengths[ix] : sample_lengths[ix]] for ix in range(n_samples)]
                # all_logprobs = [ref_logprobs[ix, prompt_lengths[ix] : sample_lengths[ix]] for ix in range(n_samples)]

 
                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=all_input_ids[sample_idx][:prompt_lengths[sample_idx]],
                        response_tensor=all_input_ids[sample_idx][prompt_lengths[sample_idx]:],
                        logprobs=ref_logprobs[:, prompt_lengths[sample_idx]-1:sample_lengths[sample_idx]-1].squeeze(),
                        values=values[:, prompt_lengths[sample_idx]-1:sample_lengths[sample_idx]-1].squeeze(),
                        rewards=rewards[sample_idx],
                    )
                )
        self.push_to_store(ppo_rl_elements)