Source code of our ICML 2024 paper "Less is More: on the Over-Globalizing Problem in Graph Transformers"

# Environment

To run our code, you should first configure your Python environment with Anaconda as follows:

```shell
conda create -n cobformer python=3.10
conda activate cobformer
conda install --yes --file requirements.txt
```

# Usage

We have provided a `run.sh` file, and you can execute the commands within it to verify our results.

To avoid the high computational cost of calculating the cluster partition for the ogbn-products dataset in each run, you can pre-calculate the cluster partition by running the `cal_partition.py` script located in the `Data` folder.
