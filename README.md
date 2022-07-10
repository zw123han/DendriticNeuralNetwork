### Dendritic Neural Networks

#### Paper
_DyNN workshop at the 39th International Conference on Machine Learning_
https://arxiv.org/abs/2207.00708

#### About
This repo contains the code and data required to replicate our work on dendritic-tree neural networks. We were inspired by the input dendrites of biological neurons, which recent neuroscience research has revealed to contain complex, non-linear computations.

#### Implementation
You can find the PyTorch implementation for a dendritic layer under `src/DendriticLayer.py`, or full classifiers using these layers under `src/Classifiers.py`.   

#### Results & Reproducibility
The experiemnt data for each type of network compared can be found under `data`. Data analysis was performed in `R` and can be found in `analysis.Rmd`.

#### Experiment setup
The `src` folder may contain artifacts from debugging and prototyping training loops. Please use the setup below for the most updated version of the experimental setup.
Colab Experiments: https://colab.research.google.com/drive/1btI3uJzI4LOC9Q6OBf67UYq_GnlxKZe9?usp=sharing

#### Citation
@article{2022parameterdendrite,  
  title={Parameter efficient dendritic-tree neurons outperform perceptrons},  
  author={Han, Ziwen and Gorobets, Evgeniya and Chen, Pan},  
  journal={arXiv preprint arXiv:2207.00708},  
  year={2022}  
}
