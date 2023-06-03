import os
from typing import Union, cast, Callable
import uuid
import json
from time import time
import ray
import numpy as np
import torch
import transformers
from rich.console import Console
from rich.table import Table
import torch.nn.functional as F

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import ILQLBatch, ILQLSeq2SeqBatch
from trlx.models.modeling_sql import (
    AutoModelForCausalLMWithSQLHeads,
    AutoModelForSeq2SeqLMWithSQLHeads,
    SQLConfig,
)
from trlx.data.accelerate_base_datatypes import PromptBatch

from trlx.pipeline.sql_on_pipeline import (
    SQLRolloutStorage,
    ILQLSeq2SeqRolloutStorage,
    tokenize_dialogue,
)
from torch.utils.data import DataLoader

from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils.modeling import RunningMoments, logprobs_of_labels

from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import to_device
from trlx.utils import Clock, infinite_dataloader
from trlx.data.ilql_types import (
    ILQLBatch,
    ILQLElement,
    ILQLSeq2SeqBatch,
    ILQLSeq2SeqElement,
)
logger = logging.get_logger(__name__)


@register_trainer
class AccelerateSQLDONTrainer(AccelerateRLTrainer):
    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        if not isinstance(config.method, SQLConfig):
            raise ValueError("config.method must be SQLConfig")

        self.ilql: SQLConfig = cast(SQLConfig, config.method)
        # Setup the rollout store
        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False
            
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        self.store = SQLRolloutStorage(self.tokenizer.pad_token_id)
        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(self.config.train.batch_size, shuffle=True)


  

        # Prepare multi-GPU acceleration
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store
        

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            # max_length=self.max_length,
            logit_mask=self.logit_mask,
            eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
        )
        
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
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std        
        
    def get_arch(self, config):
        if config.model.model_arch_type == "seq2seq":
            from_fn = AutoModelForSeq2SeqLMWithSQLHeads.from_pretrained
            if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
                from_fn = AutoModelForSeq2SeqLMWithSQLHeads.from_config
        else:
            from_fn = AutoModelForCausalLMWithSQLHeads.from_pretrained
            if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
                from_fn = AutoModelForCausalLMWithSQLHeads.from_config
        return from_fn(
            config.model.model_path,
            alpha=config.method.alpha,
        )

    def post_epoch_callback(self):
        """Post epoch callback

        Clears the store and creates `num_rollouts` new episodes.
        """
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)
        
    def post_backward_callback(self):
        if self.iter_count % self.config.method.steps_for_target_q_sync == 0:
            self.accelerator.unwrap_model(self.model).sync_target_q_heads()
            
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
            
    def loss(self, batch: Union[ILQLBatch, ILQLSeq2SeqBatch]):
        batch = to_device(batch, self.accelerator.device)
        if self.config.model.model_arch_type == "seq2seq":
            logits, qs, target_qs, _, _ = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                actions_ixs=batch.actions_ixs,
                states_ixs=batch.states_ixs,
                decoder_input_ids=batch.decoder_input_ids,
            )
        else:
            # print("input_ids", batch.input_ids, batch.input_ids.shape)
            # print("attention", batch.attention_mask, batch.attention_mask.shape)
            # print("action_ixs", batch.actions_ixs, batch.actions_ixs.shape)
            # print("states_ixs", batch.states_ixs, batch.states_ixs.shape)
            qs, target_qs, original_qs,  vs, _ = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                actions_ixs=batch.actions_ixs,
                states_ixs=batch.states_ixs,
            )

        return self.ilql.loss(qs, target_qs, original_qs, vs, batch, self.iter_count)

    def prepare_learning(self):
        # train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)
        
        self.train_dataloader = self.store.create_loader(self.config.train.batch_size, shuffle=True)
 
        # print("epochs", self.config.train.epochs)
        self.n_updates_per_batch = self.config.method.sql_epochs
        self.total_steps = self.config.train.epochs * self.n_updates_per_batch * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)
        # print("Total steps:", self.total_steps)
        
    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=True)
        prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = infinite_dataloader(prompt_dataloader)
           
    def make_experience_seq2seq(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        logger.info("Collecting rollouts")
        if self.tokenizer:
            samples = [tokenize_dialogue(s, self.tokenizer, max_length) for s in samples]

        all_input_ids = []
        all_output_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        for sample in samples:
            all_input_ids.append(torch.tensor(sample[0]))
            all_output_ids.append(torch.tensor(sample[1]))
            isoutput = False
            actions_ixs = []
            length = 0
            for phrase in sample:
                if isoutput:
                    length = len(phrase)
                    actions_ixs.append(torch.arange(0, length - 1))
                isoutput = not isoutput
            states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)

        if self.tokenizer and os.environ.get("RANK", "0") == "0":
            logger.info("Logging sample example")
            prompt = self.tokenizer.decode(all_input_ids[0])
            response = self.tokenizer.decode(all_output_ids[0])
            columns = ["Prompt", "Response", "Reward"]
            table = Table(*columns, title="Sample Example", show_lines=True)
            table.add_row(prompt, response, str(rewards[0]))
            Console().print(table)

        sample_lengths = np.array(list(map(len, all_input_ids))) + np.array(list(map(len, all_output_ids)))
        output_lengths = np.array(list(map(len, all_output_ids)))
        prompt_lengths = sample_lengths - output_lengths
        returns = torch.tensor(rewards, dtype=float)

        if os.environ.get("RANK", "0") == "0":
            logger.info("Logging experience string statistics")
            columns = ["Prompt Length", "Output Length", "Sample Length"]
            table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
            row = []
            for lengths in [prompt_lengths, output_lengths, sample_lengths]:
                row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
            table.add_row(*row)
            Console().print(table)

        returns = (returns - returns.mean()) / (returns.std() + torch.finfo(returns.dtype).eps)
        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret

        attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]
        self.store = ILQLSeq2SeqRolloutStorage(
            all_input_ids,
            attention_mask,
            all_output_ids,
            rewards,
            all_states_ixs,
            all_actions_ixs,
            all_dones,
        )

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0): # (self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """ 
        max_length = 1024

        if self.config.model.model_arch_type == "seq2seq":
            return self.make_experience_seq2seq(samples, rewards, max_length)

        logger.info("Collecting double rollouts")
        
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )
        
        sql_elements = []
        rl_elements = 0
        stats = {}
        samples_list = []
        rewards = []
        clock = Clock()
        
        def gen_responses(samples, prompt_tensors):
            device = samples.device
            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            # print("prompt", prompt_tensors)
            # print("padded prompt", padded_prompts)
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)

            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes
                )
                exp_score_time = time()
                all_scores = torch.tensor(
                    self.reward_fn(
                        samples=all_str_samples,
                        prompts=all_str_prompts,
                        outputs=all_str_outputs,
                    ),
                    dtype=torch.float,
                    device=device,
                )
                stats["time/exp_score"] = time() - exp_score_time
                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
            else:
                all_scores = None

            if torch.distributed.is_initialized():
                scores = torch.empty(len(samples), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = torch.tensor(all_scores[0])

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            

            outputs = list(map(torch.LongTensor, outputs))
            
            
            self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores)
            stats["exp_scores/mean"] = all_scores_mean
            stats["exp_scores/std"] = all_scores_std
            stats["exp_scores/running_mean"] = self.running_moments.mean
            stats["exp_scores/running_std"] = self.running_moments.std


            # if self.config.method.scale_reward == "running":
            #     scores = (scores - self.running_moments.mean) / (self.running_moments.std + 1e-10)
            # elif self.config.method.scale_reward == "ref":
            #     scores =  (scores - self.ref_mean) / (self.ref_std + 1e-10)
            return scores, outputs              
            
        while len(sql_elements) < num_rollouts:
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)
            exp_generate_time = time()
            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(**batch)
            samples_pair = self.generate(**batch)
            stats["time/exp_generate"] = time() - exp_generate_time
            prompt_tensors = batch.input_ids
            device = samples.device
            scores, outputs = gen_responses(samples, prompt_tensors)
            scores_pair, outputs_pair = gen_responses(samples_pair, prompt_tensors)
            
            fin_scores = scores - torch.log(torch.exp(scores) + torch.exp(scores_pair))
            fin_scores_pair =  scores_pair - torch.log(torch.exp(scores) + torch.exp(scores_pair))
            
            n_samples: int = samples.shape[0]

            # Estimate the KL divergence between the model and reference model
            # if self.config.model.model_arch_type == "seq2seq":
            #     attention_mask = sample_outputs != self.tokenizer.pad_token_id
            #     start = 0
            # else:
            # start = prompt_tensors.shape[1] - 1


            # prompt_tensors = prompt_tensors.cpu()
            # sample_outputs = sample_outputs.cpu()
            # values = values.cpu()[:, :-1]


            # Get the logprobs and values, for tokens that are not padding
            # or beginning of sequences tokens. These are from the model (not the reference model) 

            rollout_count = 0
            # print("start", start)
            # print("ends", ends, ends.shape)
            for sample_idx in range(n_samples):
                prompt_t = prompt_tensors[sample_idx]
                idx = torch.arange(prompt_t.shape[0], 0, -1)
                prompt_t_tomax= prompt_t.not_equal(self.tokenizer.pad_token_id).cpu() * idx.cpu()
                prompt_start = torch.argmax(prompt_t_tomax)
                prompt_t = prompt_t[prompt_start:]
                tokens = torch.cat((prompt_t.to(device), outputs[sample_idx].to(device)))
                tokens_pair = torch.cat((prompt_t.to(device), outputs_pair[sample_idx].to(device)))
                attention_mask = tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                attention_mask_pair = tokens_pair.not_equal(self.tokenizer.pad_token_id).long().to(device)
                start = prompt_t.shape[0]
                end = start + attention_mask[start:].sum() 
                end_pair = start + attention_mask_pair[start:].sum()               
                actions_ixs = []
                actions_ixs = [torch.arange(start-1, end-1)]
                actions_ixs_pair = [torch.arange(start-1, end_pair-1)]
                # print(actions_ixs[0].device)
                states_ixs = torch.hstack((*actions_ixs, torch.tensor(end).cpu()))
                dones = torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int)
                states_ixs_pair  = torch.hstack((*actions_ixs_pair, torch.tensor(end_pair).cpu()))
                dones_pair  = torch.tensor([1] * (len(states_ixs_pair) - 1) + [0], dtype=int)
                
                # print("scores", scores)
                # print("states_ixs", states_ixs, states_ixs.shape)
                # print("dones", dones, dones.shape)
                # print("actions_ixs", actions_ixs)
                # print("prompt_start", prompt_start, prompt_t, prompt_t.shape)
                # print("attention", attention_mask, attention_mask.shape)
                # all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                #     [prompt_t], [tokens], [start]
                # )
                # print("Strings:", all_str_samples, all_str_prompts, all_str_outputs)
                
                # rewards = actions_ixs[0] * 0
                # rewards[-1] += scores[sample_idx].cpu()
                if end - start == 0 or end_pair - start == 0:
                    continue
                        # returns = (returns - returns.mean()) / (returns.std() + 1e-30)
                rewards = torch.zeros(end - start)
                rewards[-1] = fin_scores[sample_idx]
                rewards_pair = torch.zeros(end_pair - start)
                rewards_pair[-1] = fin_scores_pair[sample_idx]
                # print("rewards", rewards)
                 # # print(str_prompts + str_outputs)
                # samples_list.append(str_prompts + str_outputs)
                # trans_scores = scores.cpu().numpy()
                # # print(trans_scores)
                # rewards = np.concatenate((rewards, trans_scores))
                # rl_elements += 1 
                # if fin_scores[sample_idx] > fin_scores_pair[sample_idx]:
                #     sql_elements.append(
                #         ILQLElement(
                #             tokens,
                #             attention_mask,
                #             rewards,
                #             states_ixs,
                #             actions_ixs[0],
                #             dones, 
                #         )
                #     )
                # else: 
                #     sql_elements.append(
                #         ILQLElement(
                #             tokens_pair,
                #             attention_mask_pair,
                #             rewards_pair,
                #             states_ixs_pair,
                #             actions_ixs_pair[0],
                #             dones_pair, 
                #         )
                #     )
                sql_elements.append(
                        ILQLElement(
                            tokens,
                            attention_mask,
                            rewards,
                            states_ixs,
                            actions_ixs[0],
                            dones, 
                        )
                )
                sql_elements.append(
                        ILQLElement(
                            tokens_pair,
                            attention_mask_pair,
                            rewards_pair,
                            states_ixs_pair,
                            actions_ixs_pair[0],
                            dones_pair, 
                        )
                )
                rollout_count += 2
            exp_time = clock.tick()
            tbar.set_description(f"[rollout {len(sql_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        # if torch.distributed.is_initialized():
        #     torch.distributed.all_reduce(self.mean_kl, torch.distributed.ReduceOp.AVG)

        # stats["policy/sqrt_kl"] = torch.sqrt(self.mean_kl)
        # stats["kl_ctl_value"] = self.kl_ctl.value
        stats["time/exp"] = exp_time

        if not ray.is_initialized():
            self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(sql_elements)
                        

             
        # if self.tokenizer:
        #     samples = [tokenize_dialogue(s, self.tokenizer, 1000) for s in samples_list]

        # all_input_ids = []
        # all_actions_ixs = []
        # all_states_ixs = []
        # all_dones = []
        # for sample in samples:
        #     length = 0
        #     # print(sample)
        #     # print(len(sample))
        #     # print(sum(sample, []))
        #     # print(all_input_ids)
        #     all_input_ids.append(torch.tensor(sum(sample, [])))
        #     isoutput = False
        #     actions_ixs = []
        #     for phrase in sample:
        #         if isoutput:
        #             actions_ixs.append(torch.arange(length - 1, length + len(phrase) - 1))

        #         length += len(phrase)
        #         isoutput = not isoutput

        #     states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
        #     all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
        #     all_actions_ixs.append(torch.hstack(actions_ixs))
        #     all_states_ixs.append(states_ixs)

        # if self.tokenizer and os.environ.get("RANK", "0") == "0" and iter_count<10:
        #     logger.info("Logging sample example")
        #     prompt = self.tokenizer.decode(all_input_ids[0][: all_states_ixs[0][1]])
        #     response = self.tokenizer.decode(all_input_ids[0][all_states_ixs[0][1] :])
        #     columns = ["Prompt", "Response", "Reward"]
        #     table = Table(*columns, title="Sample Example", show_lines=True)
        #     table.add_row(prompt, response, str(rewards[0]))
        #     Console().print(table)

        # sample_lengths = np.array(list(map(len, all_input_ids)))
        # output_lengths = np.array(list(map(len, all_actions_ixs)))
        # prompt_lengths = sample_lengths - output_lengths
        # returns = torch.tensor(rewards, dtype=float)

        # if os.environ.get("RANK", "0") == "0" and iter_count<10:
        #     logger.info("Logging experience string statistics")
        #     columns = ["Prompt Length", "Output Length", "Sample Length"]
        #     table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
        #     row = []
        #     for lengths in [prompt_lengths, output_lengths, sample_lengths]:
        #         row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
        #     table.add_row(*row)
        #     Console().print(table)

        # returns = (returns - returns.mean()) / (returns.std() + 1e-30)
        # rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        # for rs, ret in zip(rewards, returns):
        #     rs[-1] = ret

        # attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]

        # self.store = SQLRolloutStorage(
        #     all_input_ids,
        #     attention_mask,
        #     rewards,
        #     all_states_ixs,
        #     all_actions_ixs,
        #     all_dones,
        # )
