# Kinase-Inhibitor Binding Affinity Prediction with Pretrained Graph Encoder and Language Model

## Introduction
The accurate prediction of inhibitor-kinase binding affinity is crucial in drug discovery and medical applications, especially in the treatment of diseases such as cancer. Existing methods for predicting inhibitor-kinase affinity still face challenges including insufficient data expression, limited feature extraction, and low performance. Despite the progress made through artificial intelligence (AI) methods, especially deep learning technology, many current methods fail to capture the intricate interactions between kinases and inhibitors. Therefore, it is necessary to develop more advanced methods to solve the existing problems in inhibitor-kinase binding prediction. This study proposed Kinhibit, a novel framework for inhibitor-kinase binding affinity predictor. Kinhibit integrates self-supervised contrastive learning with multi-view molecular graph representation and structure-informed protein language model (ESM-S) to extract features effectively. Kinhibit also employed a feature fusion approach to optimize the fusion of inhibitor and kinase features. Experimental results demonstrate the superiority of this method, achieving an accuracy of 92.6% in inhibitor prediction tasks of three MAPK signaling pathway kinases: Raf protein kinase (RAF), Mitogen-activated protein kinase kinase (MEK), and Extracellular Signal-Regulated Kinase (ERK). Furthermore, the framework achieves an impressive accuracy of 93.4\% on the MAPK-All dataset. This study provides promising and effective tools for drug screening and biological sciences.

## Environment
* Anaconda3
* python 3.7.12
## Dependency
* scikit-learn   0.23.2
* pandas   1.3.5
* rdkit   2022.9.5
* torch   1.13.1
* torch-cluster   1.6.1
* torch-geometric   2.3.1
* torch-scatter   2.1.1
* torch-sparse    0.6.15
* numpy		1.26.4
* torchdrug		0.2.1
* pycaret		2.3.10
## Model Weights
We extracted features for kinases based on the structure-informed ESM ([ESM-S](https://github.com/DeepGraphLearning/esm-s)). Model weights for ESM-S can be found [here](https://huggingface.co/Oxer11/ESM-S).
The model weights for Kinhibit can be found [here](https://zenodo.org/records/15068720).
The model weights for ESM-2-650M can be found [here](https://github.com/facebookresearch/esm).
## Usage
### Contrastive Learning
```python MolGraph_Contrastive Learning.py```
### Regression model training
See Pycaret[https://github.com/pycaret/pycaret].
### Prediction(example):
```python kinhibit.py --kinase RAF1 --test_smiles test.txt --outputpath ./Results/```
