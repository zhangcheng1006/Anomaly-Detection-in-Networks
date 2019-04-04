# Anomaly-Detection-in-Networks
Course Project of Compressed Sensing

## Citation

This project is an implementation of the anomaly detection method proposed in the following paper: 

<pre><code>@article{elliott2019anomaly,
  title={Anomaly Detection in Networks with Application to Financial Transaction Networks},
  author={Elliott, Andrew and Cucuringu, Mihai and Luaces, Milton Martinez and Reidy, Paul and Reinert, Gesine},
  journal={arXiv preprint arXiv:1901.00402},
  year={2019}
}
</code></pre>

# Prerequisite
1. You need to have ```networkx```, ```rpy2``` python packages installed.
2. Following the instruction of this github repository: https://github.com/taynaud/python-louvain to install ```python-louvain``` package which performs community detection.
3. Following the instruction of this github repository: https://github.com/alan-turing-institute/network-comparison to install ```netdist``` R package which performs network comparison. And ```rpy2``` package is necessary to utilise R package in python.

# Structure
1. Under the main directory, there is a ```main.py``` to generate data under the current folder.
2. We also prepared some generated data under the directory: ```./data```.
3. Under the main directory, there is a jupyter notebook: ```demo.ipynb```, including a demo of data generation, and a demo of how to perform Anomaly Detection algorithm on the generated dataset.
