<h1 align="center">
InstructRAG 
</h1>

<h3 align="center">
Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales <br>
[<a href="https://arxiv.org/abs/2406.13629">arXiv</a>]
[<a href="https://weizhepei.com/instruct-rag-page/">Website</a>]
[<a href="https://github.com/weizhepei/InstructRAG?tab=readme-ov-file#model-checkpoints">Model</a>] [<a href="https://github.com/weizhepei/InstructRAG/tree/main/dataset">Dataset</a>]
[<a href="https://x.com/weizhepei/status/1803992285899620837">X Summary</a>]
</h3>

InstructRAG is a simple yet effective RAG framework that allows LMs to explicitly denoise retrieved contents by generating rationales for better verifiability and trustworthiness. 

![](https://weizhepei.com/instruct-rag-page/static/images/instructrag.png)

## **InstructRAG Key Features:**

- 🤖 **Self-Synthesis**: Leverage instruction-tuned LMs to generate their OWN supervision for denoising.
- 🔌 **Easy-to-Use**: Support both in-context learning (ICL) and supervised fine-tuning (SFT).
- 🚀 **Effectiveness**: Up to 8.3% better results across 5 benchmarks ([Table 3](https://arxiv.org/html/2406.13629v2#S3.T3)).
- 💪 **Noise Robustness**: Robust to increased noise ratios in various scenarios ([Figure 3](https://arxiv.org/html/2406.13629v2#S3.F3)).
- 🔁 **Task Transferability**: InstructRAG can also solve out-of-domain unseen tasks ([Figure 4](https://arxiv.org/html/2406.13629v2#S3.F4)).

Please see also our [paper](https://arxiv.org/abs/2406.13629) and [X summary](https://x.com/weizhepei/status/1803992285899620837) for more details.

## 🔗 Quick Links
- [InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales](#instructrag-key-features)
    - [Installation](#installation)
    - [Training Script](#training-script)
    - [Evaluation](#evaluation)
    - [Generation Example](#generation-example)
    - [Model Checkpoints](#model-checkpoints)

## Installation
Run the following script to create a Python virtual environment and install all required packages.
```shell
bash setup.sh
```

Alternatively, you can also directly create a conda environment using the provided configuration file.

```shell
conda env create -f environment.yml
```

## Training Script
To train the model (i.e., InstructRAG-FT), just activate the environment and run the following training script. The training config is set for 4xH100 80G GPUs. You may need to adjust NUM_DEVICE and PER_DEVICE_BATCH_SIZE based on your computation environment.

```shell
conda activate instrag
bash train.sh
```
## Evaluation
There are two instantiations of our framework:
- InstructRAG-ICL: training-free & easy-to-adapt
- InstructRAG-FT: trainable & better performance

Use the following script to evaluate InstructRAG in both training-free and trainable settings. You can specify the task and model by adjusting DATASET and MODEL in `eval.sh`.

```shell
conda activate instrag
bash eval.sh
```


## Generation Example

The following case study shows that InstructRAG can effectively identify relevant information from noisy input and leverage its own knowledge to correctly answer questions when required. The red texts denote irrelevant or inaccurate model generations, while the green texts denote contents relevant to the question. 

![](https://weizhepei.com/instruct-rag-page/static/images//case_study.png)

## Model Checkpoints
Below is the full list of InstructRAG models fine-tuned on each dataset in our work.

| Dataset | HF Model Repo | Retriever |
|------------------------------|-----------------------------------------------------------------------------------------------------------|:------:|
| PopQA | [meng-lab/PopQA-InstructRAG-FT](https://huggingface.co/meng-lab/PopQA-InstructRAG-FT) | Contriever |
| TriviaQA | [meng-lab/TriviaQA-InstructRAG-FT](https://huggingface.co/meng-lab/TriviaQA-InstructRAG-FT) | Contriever |
| Natural Questions | [meng-lab/NaturalQuestions-InstructRAG-FT](https://huggingface.co/meng-lab/NaturalQuestions-InstructRAG-FT) | DPR |
| ASQA | [meng-lab/ASQA-InstructRAG-FT](https://huggingface.co/meng-lab/ASQA-InstructRAG-FT) | GTR |
| 2WikiMultiHopQA | [meng-lab/2WikiMultiHopQA-InstructRAG-FT](https://huggingface.co/meng-lab/2WikiMultiHopQA-InstructRAG-FT) | BM25 |

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zhepei (zhepei.wei@virginia.edu). If you encounter any problems when using the code, or want to report a bug, feel free to open an issue! Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@inproceedings{
wei2025instructrag,
title={Instruct{RAG}: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales},
author={Zhepei Wei and Wei-Lin Chen and Yu Meng},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=P1qhkp8gQT}
}
```
