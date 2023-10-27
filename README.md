This repository contains the source code and data corresponding to PLOS 2023 publication:

## Population-wise Labeling of Sulcal Graphs using Multi-graph Matching.

The graph matching methods used in this work were implemented using authors implementation and can be found in their corresponding repositories:
#### [Kernelized graph matching](https://proceedings.neurips.cc/paper_files/paper/2019/file/cd63a3eec3319fd9c84c942a08316e00-Paper.pdf) ([Code](https://github.com/ZhenZhang19920330/KerGM_Code))

#### [Multi-image matching via fast alternating minimization](https://arxiv.org/pdf/1505.04845.pdf) ([Code](https://github.com/zju-3dv/multiway))
#### [Multi-Graph Matching via Affinity Optimization with Graduated Consistency Regularization](https://faculty.cc.gatech.edu/~zha/papers/TPAMI2477832_V2.pdf) ([Code](https://github.com/Thinklab-SJTU/pygmtools))
#### [Solving the multi-way matching problem by permutation synchronization](https://pages.cs.wisc.edu/~pachauri/perm-sync/assignmentsync.pdf) ([Code](https://pages.cs.wisc.edu/~pachauri/perm-sync))

----------------------------------------------------------------------------------------------------------

#### All permutation matrices, silhouette values and consistency values are precomputed and provided in the folder `Oasis_original_new_with_dummy`

### Dependencies:

These dependencies are mandatory for the implementation of scripts provided, please follow the instructions properly provided in the following dependencies for proper execution of the scripts provided:

#### 1. [SLAM](https://github.com/gauzias/slam)
#### 2. [VISBRAIN](https://github.com/EtienneCmb/visbrain) for visualization.


### To generate a population of synthetic sulcal graphs(specify the parameters in script):
`python script_generation_graphs_with_edges_permutation_updated.py`


### Computation of affinity matrices:
`python script_generation_affinity_and_incidence_matrix.py`
