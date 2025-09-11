# mogt
MOGT is a semi-supervised graph neural network that integrates the multi-omics data of genes and the topology information of biological networks.To construct the multi-omics representation graph, we represented genes as nodes and SNP-SNP interactions as edges. The node features were the multi-omics data of genes, including differential expression (DE), enhancer-promoter interactions (EPI) in patients and controls, and gene expression in five brain regions（Parietal Lobe, Frontal Lobe, Temporal Lobe, Cerebellum, and occipital Lobe）in adolescents and adults.

MOGT was written in Python 3.8, and should run on any OS that support pytorch and pyg. Training is faster on a GPU with at least 24G memory to reproduce our results.

## Documentation
MOGT documentation is available through [Documentation](https://sunyolo.github.io/CGMega.github.io/).

## Conda Environment
We recommend using conda to configure the code runtime environment:
```
conda create -n MOGT python=3.8.20
```
MOGT has the following dependencies:
```
matplotlib==3.7.2
model==0.6.0
numpy==1.24.3
pandas==2.0.3
PyYAML==6.0.2
PyYAML==6.0.2
scikit_learn==1.3.2
torch==1.11.0
torch_geometric==2.0.4
transformers==4.46.3
utils==1.0.2
wandb==0.21.3
```
Install commmands of torch_scatter and torch_sparse should be adjusted according to pytorch and cuda version, see [PyG 2.0.3 Installation](https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/installation.html)

## Installation
We recommend getting MOGT using Git from our Github repository through the following command:

```
git clone 
```
To verify a successful installation, just run:
```
python main.py -cv -l  # Importing the relevant libraries may take a few minutes.
```

## Tutorial
This tutorial demonstrates how to use MOGT functions with a demo dataset (SCZ as an example). 
Once you are familiar with MOGT’s workflow, please replace the demo data with your own data to begin your analysis. 
[Tutorial notebook](https://github.com/NBStarry/CGMega/tree/main/Tutorial.ipynb) is available now.

### How to prepare input data

We recommend getting started with MOGT using the provided demo dataset. When you want to apply MOGT to your own multi-omics dataset, please refer to the following tutorials to learn how to prepare input data.

Overall, the input data consists of two parts: the graph, constructed from SNP-SNP interaction and the node feature including DE, EPI, and gene expression in five brain regions（Parietal Lobe, Frontal Lobe, Temporal Lobe, Cerebellum, and Occipital Lobe）in adolescents and adults.

 If you are unfamiliar with MOGT, you may start with our data used in the paper to save your time. For SCZ, the input data as well as the label information are uploaded [here](https://github.com/NBStarry/CGMega/tree/main/data). If you start with this data, you can skip the _step 1_ about _How to prepare input data_.
 The following steps from 1.1~1.3 can be found in our source code [data_preprocess_cv.py](https://github.com/NBStarry/CGMega/blob/main/data_preprocess_cv.py).

>The labels should be collected yourself if you choose analyze your own data.


Please choose parameters by [NeoLoopFinder](https://github.com/XiaoTaoWang/NeoLoopFinder) to suit your data. An example is available in [batch_neoloop.sh](https://github.com/NBStarry/CGMega/blob/main/data/AML_Matrix/batch_neoloop.sh)

---


### MOGT framework

![image]()

---

### Hyperparameters

- To reduce the number of parameters and make training feasible within time and resource constraints, the input graphs were sampled using neighbor sampler. The subgraphs included all first and second order neighbors for each node and training was performed on these subgraphs.
- The learning rate is increased linearly from 0 to 0.005 for the first 20% of the total iterations.
- warm-up strategy for learning rate is employed during the initial training phase.
- To prevent overfitting and over smoothing, an early stop strategy is adopted. If the model's performance on the validation set dose not improve for a consecutive 100 epochs, the training process stops.
- Dropout is used and the dropout rate is set to 0.1 for the attention mechanism and 0.4 for the other modules.
- Max pooling step size is 2. After pooling, the representation had 192 dimensions.

---

### System and Computing resources

| Item       | Details          |
| ---------- | ---------------- |
| System     | Ubuntu 20.04.1 LTS |
| RAM Memory | 256G               |
| GPU Memory | NVIDIA GeForce RTX, 24G     |
| Time       | ~ 30m             |

```note
The above table reports our computing details during CGMega development and IS NOT our computing requirement.

If your computer does not satisfy the above, you may try to lower down the memory used during model training by reduce the sampling parameters, the batch size or so on. 
.

```

## Questions and Code Issues
If you are having problems with our work, please use the [Github issue page](https://github.com/NBStarry/CGMega/issues).