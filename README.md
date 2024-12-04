# Lightning-Template

[![python](https://img.shields.io/badge/-Python_3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white&style=flat-square)](https://github.com/tshu-w/lightning-template)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4+-ee4c2c?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org)
[![lightning](https://img.shields.io/badge/Lightning_2.4+-792ee5?logo=pytorchlightning&logoColor=white&style=flat-square)](https://lightning.ai)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray&style=flat-square)](https://github.com/tshu-w/lightning-template?tab=MIT-1-ov-file)

A clean and flexible [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) template to kickstart and structure your deep learning project, ensuring efficient workflow, reproducibility, and easy extensibility for rapid experiments.

### Why Lightning-Template?

Pytorch Lightning is a deep learning framework designed for professional AI researchers and engineers, freeing users from boilerplate code (_e.g._, multiple GPUs/TPUs/HPUs training, early stopping, and checkpointing) to focus on going from idea to paper/production.

This Lightning template leverages [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) to separate configuration from source code, guaranteeing reproducibility of experiments, and incorporates many other [best practices](#best-practices).

+ **Compared to [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)**: Our template provides similar functionality through a simple and straightforward encapsulation of Lightning's built-in CLI, making it suitable for users who prefer minimal setup without an additional Hybra layer.

> Note: This is an unofficial project that lacks comprehensive test and continuous integration.

### Quickstart

```console
git clone https://github.com/YourGithubName/your-repository-name
cd your-repository-name

# [SUGGESTED] use conda environment
conda env create -n env-name -f environment.yaml
conda activate env-name

# [ALTERNATIVE] install requirements directly
pip install -r requirements.txt

# Run the sample script, i.e., ./run fit --config configs/mnist.yaml
bash -x scripts/run.sh
```

### Workflow - how it works

Before using this template, please read the basic Pytorch Lightning documentation: [Lightning in 15 minutes](https://lightning.ai/docs/pytorch/stable/starter/introduction.html).

1. Define a [Lightning Module](https://lightning.ai/docs/pytorch/2.4.0/common/lightning_module.html) (Examples: [mnist_model.py](src/models/mnist_model.py) and [glue_transformer.py](src/models/glue_transformer.py))
2. Define a [Lightning DataModule](https://lightning.ai/docs/pytorch/2.4.0/data/datamodule.html#lightningdatamodule) (Examples: [mnist_datamodule.py](src/datamodules/mnist_datamodule.py) and [glue_datamodule.py](src/datamodules/glue_datamodule.py))
3. Prepare your experiment configs (Examples: [mnist.yaml](configs/mnist.yaml) and [mrpc.yaml](configs/mrpc.yaml))
4. Run experiments (_cf._,  [Configure hyperparameters from the CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html))
   - To see the available commands type:
   ```console
   ./run --help
   ```
   - Train a model from the config:
   ```console
   ./run fit --config configs/mnist.yaml
   ```
   - Override config options:
   ```console
   ./run fit --config configs/mnist.yaml --trainer.precision 16 --model.learning_rate 0.1 --data.batch_size 64
   ```
   - Separate model and datamodule configs:
   ```console
   ./run fit --config configs/data.yaml --config configs/model.yaml
   ```

### Project Structure
The directory structure of a project looks like this:
```
lightning-template
├── configs               ← Directory of Configs
│   ├── mnist.yaml
│   ├── mrpc.yaml
│   ├── presets           ← Preset configs for Lightning features
│   └── sweep_mnist.yaml
├── data                  ← Directory of Data
├── environment.yaml
├── models                ← Directory of Models
├── notebooks             ← Directory of Notebooks
├── pyproject.toml
├── README.md
├── requirements.txt
├── results               ← Directory of Results
├── run                   ← Script to Run Lightning CLI
├── scripts               ← Directory of Scripts
│   ├── print_results
│   ├── run.sh
│   ├── sweep             ← Script to sweep Experiments
│   └── sweep_mnist.sh
└── src                   ← Directory of Source Code
    ├── callbacks
    ├── datamodules
    ├── models
    ├── utils
    └── vendor            ← Directory of Third-Party Code
```

### Best Practices
1. Use [conda](https://docs.anaconda.com/miniconda/) to manage environments.
2. Leverages Lightning awesome features (_cf._, [How-to Guides](https://lightning.ai/docs/pytorch/stable/common/) & [Glossary](https://lightning.ai/docs/pytorch/stable/glossary/))
3. Use [pre-commit](https://pre-commit.com) and [ruff](https://docs.astral.sh/ruff) to check and format code with configuration in [pyproject.toml](pyproject.toml) and [.pre-commit-config.yaml](.pre-commit-config.yaml).
   ```console
   pre-commit install
   ```
4. Use [dotenv](https://github.com/motdotla/dotenv) to automatically change environments and set variables (_cf._, [.envrc](.envrc)).
   ```console
   λ cd lightning-template
   direnv: loading ~/lightning-template/.envrc
   direnv: export +CONDA_DEFAULT_ENV +CONDA_EXE +CONDA_PREFIX +CONDA_PROMPT_MODIFIER +CONDA_PYTHON_EXE +CONDA_SHLVL +_CE_CONDA +_CE_M ~PATH ~PYTHONPATH
   ```
   1. Add the project root to `PATH` to use `run` script directly.
   ```console
   export PATH=$PWD:$PWD/scripts:$PATH
   run fit --config configs/mnist.yaml
   ```
   2. Add the project root to `PYTHONPATH` to avoid modifying `sys.path` in scripts.
   ```console
   export PYTHONPATH=$PWD${PYTHONPATH:+":$PYTHONPATH"}
   ```
   3. Save privacy variable to `.env`.
5. Use [shtab](https://jsonargparse.readthedocs.io/en/stable/#tab-completion) to generate shell completion file.
   <img width="100%" alt="Screenshot 2024-08-16 at 22 57 14" src="https://github.com/user-attachments/assets/70c4adfe-d587-4624-9012-31141c3748b2">
6. Use [ray tune](https://docs.ray.io/en/latest/tune/index.html) to sweep parameters or hyperparameter search (_cf._, [sweep_cli.py](src/utils/sweep_cli.py)).
   ```console
   bash ./scripts/sweep --config configs/sweep_mnist.yaml
   ```
7. Use third-party logger (_e.g._, [w&b](https://wandb.ai) and [aim](https://aimstack.io)) to track experiments.

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT

---

<div align="center">

<h2 id="your-project-name">Your Project Name</h2>

<p>
<a href="https://arxiv.org/abs/1706.03762"><img src="http://img.shields.io/badge/arXiv-1706.03762-B31B1B.svg?style=flat-square" alt="Arxiv" /></a>
<a href="https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"><img src="http://img.shields.io/badge/NeurIPS-2017-4b44ce.svg?style=flat-square" alt="Conference" /></a>
</p>

</div>

## Description
What it does

## How to run
First, install dependencies
```console
# clone project
git clone https://github.com/YourGithubName/your-repository-name
cd your-repository-name

# [SUGGESTED] use conda environment
conda env create -f environment.yaml
conda activate lit-template

# [ALTERNATIVE] install requirements directly
pip install -r requirements.txt
```

Next, to obtain the main results of the paper:
```console
# commands to get the main results
```

You can also run experiments with the `run` script.
```console
# fit with the demo config
./run fit --config configs/demo.yaml
# or specific command line arguments
./run fit --model MNISTModel --data MNISTDataModule --data.batch_size 32 --trainer.gpus 0

# evaluate with the checkpoint
./run test --config configs/demo.yaml --ckpt_path ckpt_path

# get the script help
./run --help
./run fit --help
```

## Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
