import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def calculate_confidence_interval(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)
    z = 1.96  # Z-value at 95% confidence level
    bound = (z * std / np.sqrt(n))
    return bound



def get_gene_list(rename=False,disease = "SCZ"):
    filename = "data/"+disease + "/" + disease + ".csv"
    gene_list = pd.read_csv(filename)
    gene_list = gene_list[["gene_name","gene_id"]]
    return gene_list.rename(columns={'gene_name' : 'Gene Name'}) if rename else gene_list
