# Neural Catalog: Scaling Species Recognition with Catalog of Life-Augmented Generation

This repository contains the code for the paper ["Neural Catalog: Scaling Species Recognition with Catalog of Life-Augmented Generation"](https://arxiv.org/abs/2505.05635).

## Features

Pre-extracted features used in this work can be downloaded from [here](https://waga.s3.us-west-2.amazonaws.com/feats.zip).

## Running

The provided code can be run using

```
python inf.py --help
usage: VR-RAG Inference [-h]
            [--dataset {birdsnap,cub,inat,indian,nabirds}]
```

- Example 1: to run VR-RAG on CUB

```
python inf.py --dataset cub
```

## Data: The data folder contains all the test splis from the five benchmarks. 
* CUB: [Download from here](https://www.vision.caltech.edu/datasets/cub_200_2011/)
* iNaturalist: [Download from here](https://github.com/visipedia/inat_comp/tree/master/2021)
* Birdsnap: [Download from here](https://huggingface.co/datasets/sasha/birdsnap)
* NABirds: [Download from here](https://dl.allaboutbirds.org/nabirds)
* Indian Birds: [Download from here](https://www.kaggle.com/datasets/ichhadhari/indian-birds)

## Citation

```
@misc{khan2025neuralcatalogscalingspecies,
      title={Neural Catalog: Scaling Species Recognition with Catalog of Life-Augmented Generation}, 
      author={Faizan Farooq Khan and Jun Chen and Youssef Mohamed and Chun-Mei Feng and Mohamed Elhoseiny},
      year={2025},
      eprint={2505.05635},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.05635}, 
}
```