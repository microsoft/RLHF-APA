# Fine-tuning language models with Advantage-Induced Policy Alignment
This repo contains the official implementation of paper "Fine-tuning language models with Advantage-Induced Policy Alignment", by Banghua Zhu, Hiteshi Sharma, Felipe Vieira Frujeri, Shi Dong, Chenguang Zhu, Michael I. Jordan, Jiantao Jiao.


### Abstract
Reinforcement learning from human feedback (RLHF) has emerged as a reliable approach to aligning large language models (LLMs) to human preferences. Among the plethora of RLHF techniques, proximal policy optimization (PPO) is of the most widely used methods. Despite its popularity, however, PPO may suffer from mode collapse, instability, and poor sample efficiency. We show that these issues can be alleviated by a novel algorithm that we refer to as Advantage-Induced Policy Alignment (APA), which leverages a squared error loss function based on the estimated advantages. We demonstrate empirically that APA consistently outperforms PPO in language tasks by a large margin, when a separate reward model is employed as the evaluator.
In addition, compared with PPO, APA offers a more stable form of control over the deviation from the model's initial policy, ensuring that the model improves its performance without collapsing to deterministic output.
In addition to empirical results, we also provide a theoretical justification supporting the design of our loss function.


### Getting Started
Python 3 is required for the current codebase. It's recommended to use Python 3.9 for installing the dependencies. Due to the current [Ray support issue](https://github.com/ray-project/ray/issues/33232), Python 3.11 may give error during executation.

Install the dependencies as follows.

```shell 
pip install -r requirements.txt 
pip install -e . 
```

To reproduce the experiments in the paper, execute the following set of code:
```shell 
## For running APA or AWR on HH dataset
accelerate launch  --config_file configs/accelerate/zero2-bf16.yaml examples/hh/sppo_hh.py 

## For running PPO on HH dataset
accelerate launch  --config_file configs/accelerate/zero2-bf16.yaml examples/hh/ppo_hh.py 

## For running APA or AWR on TLDR dataset
accelerate launch  --config_file configs/accelerate/zero2-bf16.yaml examples/hh/sppo_tldr.py 

## For running PPO on TLDR dataset
accelerate launch  --config_file configs/accelerate/zero2-bf16.yaml examples/hh/ppo_tldr.py 

## For running offline ILQL on HH dataset
accelerate launch  --config_file configs/accelerate/zero2-bf16.yaml examples/hh/ilql_hh.py 

## For running offline APA or AWR on HH dataset
accelerate launch  --config_file configs/accelerate/zero2-bf16.yaml examples/hh/sppo_off_hh.py  
```

Inside each of the code file, one may adjust the random seed, model size and algorithm. Note that this code is not optimized with memory usage, only for a preliminary illustration of the differences between the existing policy iteration algorithms for RLHF. The code is tested on 4 V100 and 8 V100 for 125M and 1B models, and 4 MI200 for 6B models. We put reference model, reward model and value model in three difference GPUs. For smaller number of GPUs, you may need to change the device number in accelerate_sppo_trainer.py (and other corresponding accelerator files). 


### Acknowledgement
Our codebase is built based on a stable version of [CarperAI/trlX](https://github.com/CarperAI/trlx). We thank the authors for the nicely organized code!


### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
