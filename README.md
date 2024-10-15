# PDHG-Unrolled Learning-to-Optimize Method for Large-Scale Linear Programming

## Introduction

Solving large-scale linear programming (LP) problems is an important task in various areas such as communication networks, power systems, finance and logistics. Recently, two distinct approaches have emerged to expedite LP solving: (i) First-order methods (FOMs); (ii) Learning to optimize (L2O). In this work, we propose an FOM-unrolled neural network (NN) called PDHG-Net, and propose a two-stage L2O method to solve large-scale LP problems. The new architecture PDHG-Net is designed by unrolling the recently emerged PDHG method into a neural network, combined with channel-expansion techniques borrowed from graph neural networks. We prove that the proposed PDHG-Net can recover PDHG algorithm, thus can approximate optimal solutions of LP instances with a polynomial number of neurons. We propose a two-stage inference approach: first use PDHG-Net to generate an approximate solution, and then apply PDHG algorithm to further improve the solution. Experiments show that our approach can significantly accelerate LP solving, achieving up to a 3× speedup compared to FOMs for large-scale LP problems.
**This work has beend accepted by ICML 2024**

## Usage

All codes are inside the src folder

### Training/Testing on IP/LB dataset: src_iplb
1. Generate instance using **genIpsol.py/genLbsol.py** to generate training samples. Detailed args can be found in the code.
2. Training the model using **training_ip.py/training_ip_small.py/training_lb.py/training_lb_small.py**.
3. Test the performance using **test-packing-ip.py/test-packing-lb.py/test-packing-lbsmall.py**. Log files will be stored at designated directories.

### Training/Testing on Pagerank dataset: src_pagerank
1. Generate instance using **pagerank111_gen.py** to generate training samples. Detailed args can be found in the code.
2. Training the model using **pagerank222_train.py** or **pagerank222_train_save_memory.py** for better memory management.
3. Test the performance using **test-packing.py**. Log files will be stored at designated directories.

## Citation

If you would like to use this repository, please consider citing this work using the following Bibtex:
@misc{li2024pdhgunrolledlearningtooptimizemethodlargescale,
      title={PDHG-Unrolled Learning-to-Optimize Method for Large-Scale Linear Programming}, 
      author={Bingheng Li and Linxin Yang and Yupeng Chen and Senmiao Wang and Qian Chen and Haitao Mao and Yao Ma and Akang Wang and Tian Ding and Jiliang Tang and Ruoyu Sun},
      year={2024},
      eprint={2406.01908},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.01908}, 
}

# TODO:: finialize and organize code
