# TDTDecompsition
This repository contains the source code of the paper "TDT: Tensor based Directed Truss Decomposition" by Guojing Li, Prof. Yuanyuan Zhu, Junchao Ma, Prof. Ming Zhong, Prof. Tieyun Qian, and Prof. Jeffrey Xu Yu.

## Overview
We provide source codes of truss decomposition、k-truss search、and triangle counting based on tensor, along with the installation package of extended tensor operator library `trusstensor`.

The code of tenser based algorithms are implemented by Python in the PyTorch framework. The extended tensor operator library `trusstensor` is implemented by C++ and CUDA at a lower lever.

## Experimental environments
The operating system is Ubuntu 20.04, and development tools such as g++ 9.4.0, Python 3.8, PyTorch 2.0.0, torch-scatter 2.1.2, and CUDA 11.8 are installed to ensure that the test environment can fully utilize next-generation hardware acceleration technologies,
supporting the efficient testing and development of truss
decomposition algorithms.

## Dataset
The datasets are sourced from well-known platforms such as
[SNAP (Stanford Network Analysis Platform)](https://snap.stanford.edu/data/) and [the Network Repository](https://networkrepository.com/index.php).

## Algorithms running 
### Installation package of extended tensor operator library `trusstensor`

```
python ./TDTDecomposition/demo_truss/hpu_extension/setup.py install
```

### Truss decomposition running

- Optimized tensor based directed truss decomposition
    ```
    python  /root/autodl-tmp/TDTDecomposition/demo_truss/paperexperiental.py --graph /root/autodl-tmp/data/clueweb.ungraph.mtx  --output /root/autodl-tmp/output/clueweb.pth --cuda
  ```

- Fusion experiments
    ```
    python  /root/autodl-tmp/TDTDecomposition/demo_truss/paperexperiental.py --graph /root/autodl-tmp/data/clueweb.ungraph.mtx  --output /root/autodl-tmp/output/clueweb.pth --cuda
  ```
