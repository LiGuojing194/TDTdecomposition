# TDTDecompsition
This repository contains the source code of the paper "TDT: Tensor based Directed Truss Decomposition" by Guojing Li, Prof. Yuanyuan Zhu, Junchao Ma, Prof. Ming Zhong, Prof. Tieyun Qian, and Prof. Jeffrey Xu Yu.

## Overview
We provide source codes of truss decomposition、k-truss search、and triangle counting based on tensor, along with the installation package of extended tensor operator library `trusstensor`.

The code of tenser based algorithms are implemented by Python in the PyTorch framework. The extended tensor operator library `trusstensor` is implemented by C++ and CUDA at a lower lever.

## Experimental environments
The operating system is Ubuntu 20.04, and development tools such as g++ 9.4.0, Python 3.8, PyTorch 2.0.0, torch-scatter 2.1.2, and CUDA 11.8 are installed to ensure that the test environment can fully utilize next-generation hardware acceleration technologies, supporting the efficient testing and development of truss
decomposition algorithms.

## Dataset
The datasets are sourced from well-known platforms such as
[SNAP (Stanford Network Analysis Platform)](https://snap.stanford.edu/data/) and [the Network Repository](https://networkrepository.com/index.php).

## Algorithms running 
#### ***Installation package of extended tensor operator library `trusstensor`***
Modify the absolute path of the sources parameter in `setup.py`, then modify the path in the following command line and run to install the `trusstensor` library. Note that the `extra_compile_args` parameter in `setup.py` should be configured according to your experimental environment. For instance, when using NVIDIA A100 GPUs, you need to specify the corresponding compute capability by modifying the gencode flag to `-gencode=arch=compute_80,code=sm_80`.
```
python  install /root/autodl-tmp/TDTdecomposition/demo_truss/myops/mysrc/hpu_extension/setup.py install
```

####  ***Truss decomposition***
Modify the project path in the header of all python files to be run.
```
sys.path.append('/root/autodl-tmp/TDTdecomposition')
```

- **Optimized tensor based directed truss decomposition**

    Modify the absolute path  and run the following command for optimized TDT decomposition.
    ```python
    python /root/autodl-tmp/TDTdecomposition/demo_truss/singlegpu_truss.py  --graph /root/autodl-tmp/TDTdecomposition/test_data/example_graph.txt  --output  /root/autodl-tmp/TDTdecomposition/test_data/output/test.pth  --cuda
    ```
    The `read_prepro_save(args)` function is run to preprocess the graph into a directed data structure for truss decomposition , which exists in the xxx.pth file. After that, the xxx.pth file is loaded directly to perform the computation.

- **Fusion Experiment**

    Run the `fusionexperiental.py` script by the following command.
    
    ```python
    python /root/autodl-tmp/TDTdecomposition/demo_truss/fusionexperiental.py  --graph /root/autodl-tmp/TDTdecomposition/test_data/example_graph.txt  --output  /root/autodl-tmp/TDTdecomposition/test_data/output/test.pth  --cuda
    ```
    The support computation and update functions before optimization are support_computing_before() and all_affect_support_not_before(), respectively; the optimized ones are support_computing() and all_affect_support_not(), respectively.