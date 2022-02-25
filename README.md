# Lightning-Template

A clean and scalable template to structure ML paper-code the same so that work can easily be extended and replicated.

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT

---

<div align="center">

# Your Project Name

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->


<!--
Conference
-->
</div>

## Description
What it does

## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/YourGithubName/your-repository-name
cd your-repository-name

# [OPTIONAL] create conda environment
conda create -n template python=3.9
conda activate template

# install requirements
pip install -r requirements.txt
```

Next, run experiments with the `run` script.
```bash
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
