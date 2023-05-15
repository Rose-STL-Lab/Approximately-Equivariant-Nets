## Paper: 
Rui Wang*, Robin Walters*, Rose Yu; [Approximately Equivariant Networks for Imperfectly Symmetric Dynamics](https://arxiv.org/abs/2201.11969); International Conference on Machine Learning (ICML) 2022

## Abstract:
Incorporating symmetry as an inductive bias into neural network architecture has led to improvements in generalization, data efficiency, and physical consistency in dynamics modeling. Methods such as CNN or equivariant neural networks use weight tying to enforce symmetries such as shift invariance or rotational equivariance. However, despite the fact that physical laws obey many symmetries, real-world dynamical data rarely conforms to strict mathematical symmetry either due to noisy or incomplete data or to symmetry breaking features in the underlying dynamical system. We explore approximately equivariant networks which are biased towards preserving symmetry but are not strictly constrained to do so. By relaxing equivariance constraints, we find that our models can outperform both baselines with no symmetry bias and baselines with overly strict symmetry in both simulated turbulence domains and real-world multi-stream jet flow.

## Description of Files
1. data_prep.py: code for generating PhiFlow translation, rotation and scaling datasets.

2. models: pytorch implementation of all non-equivariant, equivariant and approximately equivariant models.
     
3. run_model.py: model training and evaluation.

4. utils.py: pytorch dataset function and training helper functions.

## Requirements
- To install requirements
```
pip install -r requirements.txt
```

## Instructions
### Dataset and Preprocessing
- Install PhiFlow First 
```
git clone -b 2.0.1 --single-branch https://github.com/tum-pbs/PhiFlow.git
```

- Move data_prep.ipynb to the PhiFlow folder and run data_prep.ipynb to generate approximate translation, rotation and scaling symmetry smoke plume datasets.

- Unfortunately, we can't release the Jetflow dataset due to policy

### Training
- Train relaxed translation group convoluton, relaxed rotation steeratble CNNs and relaxed scaling steeratble CNNs.
```
sh run.sh
```

## Cite
```
@inproceedings{wang2022approximately,
title={Approximately Equivariant Networks for Imperfectly Symmetric Dynamics},
author={Rui Wang and Robin Walters and Rose Yu},
booktitle={International Conference on Machine Learning},
year={2022},
organization={PMLR}
}
```