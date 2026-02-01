# PLAN: Federated Domain Generalization via Prompt Learning and Aggregation
The official code for the paper ***Federated Domain Generalization via Prompt Learning and Aggregation***.
Accepted by IEEE Transactions on Information Forensics and Security.


 ![PLAN](/figures/PLAN.png)

**Abstract:** Federated domain generalization (FedDG) aims to improve the global model’s generalization in unseen domains by addressing data heterogeneity under privacy-preserving constraints. A common strategy in existing FedDG studies involves sharing domain-specific knowledge among clients, such as spectrum information, class prototypes, and data styles.  However, this knowledge is extracted directly from local client samples, and sharing such sensitive information poses a potential risk of data leakage, which might not fully meet the requirements of FedDG. In this paper, we introduce prompt learning to adapt pre-trained vision-language models (VLMs) in the FedDG scenario, and leverage locally learned prompts as a more secure bridge to facilitate knowledge transfer among clients. Specifically, we propose a novel FedDG framework through Prompt Learning and AggregatioN (PLAN), which comprises two training stages to collaboratively generate local prompts and global prompts at each federated round. First, each client performs both text and visual prompt learning using their own data, with local prompts indirectly synchronized by regarding the global prompts as a common reference. Second, all domain-specific local prompts are exchanged among clients and selectively aggregated into the global prompts using lightweight attention-based aggregators.
The global prompts are finally applied to adapt VLMs to unseen target domains. As our PLAN framework requires training only a limited number of prompts and lightweight aggregators, it offers notable advantages in computational and communication efficiency for FedDG. Extensive experiments demonstrate the superior generalization ability of PLAN across four benchmark datasets. 

### Visualization

![CAMs](/figures/CAMs.jpg)

## Requirement

The required packages are listed in `requirements.txt` for minimum requirement (Python 3.8.5):

```
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
$ pip install -r requirements.txt
$ pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## Datasets

[
PACS](https://wjdcloud.blob.core.windows.net/dataset/PACS.zip).

[VLCS](https://wjdcloud.blob.core.windows.net/dataset/VLCS/VLCS.zip).

[Office-Home](https://wjdcloud.blob.core.windows.net/dataset/OfficeHome.zip)

[DomainNet](https://ai.bu.edu/M3SDA/)  (Download the **cleaned** version of split files  for DomainNet dataset.)

## Results on Office-Home

| Methods         | Art       | Clipart   | Product   | Real      | Average   |
| --------------- | --------- | --------- | --------- | --------- | --------- |
| FedCLIP         | 78.45     | 64.77     | 87.68     | 87.84     | 79.69     |
| PromptFL        | 82.98     | 68.98     | 92.14     | 90.27     | 83.59     |
| FedMaPLe        | 84.56     | 72.82     | 92.38     | 91.07     | 85.21     |
| **PLAN (Ours)** | **86.65** | **74.73** | **93.47** | **92.06** | **86.73** |



## How to run

We provide the commands for four tasks in Office-Home to reproduce the results.

```
 python methods/PLAN.py --dataset office-home --mode FedAtImg --test_envs 0 --iters 6   --wk_iters 1 --num_shots 0  --root_dir DATA_PATH --text_embedding_path TEXT_EMB_PATH --batch 32 --N_CTX 8 --lr2 0.0015 --c 1.0
```



## Acknowledgements

[FedCLIP](https://github.com/microsoft/PersonalizedFL.)

[MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)
