# INQ

[Xavier Andrade](mailto:xavier@llnl.gov) (LLNL), [Alfredo A. Correa](mailto:correaa@llnl.gov) (LLNL)

INQ is an engine for electronic structure calculations.
It can work in three ways, as a standalone electronic structure code, as a library to implement complex electronic structure methods, or as a proxy-app to evaluate the performance of electronic structure algorithms in high-performance computing platforms.
It concentrates on algorithms as a portability hardware layer, generic types, and flexibility of usage.

INQ is based on density functional theory (DFT).
It can calculate ground state properties in DFT, and also excited states using time-dependent DFT (TDDFT) in real-time and linear response.
It implements different approximations for the exchange and correlation part: semi-local functionals like LDAs, GGAs, and metaGGAs, and hybrid functionals that are implemented through the ACE approach for fast execution.

One key feature of INQ is that is designed from the ground up to work in modern high-performance computing platforms.
It has support for GPU parallelization (through different frameworks), thread parallelization, and distributed memory parallelization using MPI (and Nvidia NCCL).
The thread parallelization is designed so that different tasks within the DFT approach are performed simultaneously, this achieves better scalability with respect to data parallelization alone.
INQ can perform calculations with unpolarized, polarized, and non-colinear spin.

INQ attempts to be as agnostic as possible with the representation used for the states and other quantities that appear in DFT.
It uses plane waves by default, but other representations like real space will be available.

To keep the code simple, INQ is designed as modular as possible, it uses libraries for most of the operations that can be done independently.
Some libraries, like pseudopod (for pseudopotential parsing) or multi (for multidimensional arrays) are developed by INQ authors.
Others are written by third parties, like libxc (for exchange-correlation functionals).
In the future, some other parts of INQ might be split into an independent library.

This is a list of libraries INQ depends on:

* boost-multi: for multidimensional arrays (developed by A. A. Correa)
* pseudopod: for pseudopotential parsing (developed X. Andrade)
* boost-mpi3: for distributed memory parallelization (developed by A. A. Correa)
* libxc: for exchange correlation functionals (X. Andrade is a contributor)
* cuda: for GPU parallelization
* pcg-cpp: for random number generation
* catch2: for unit testing and validation
* blas/lapack: for linear algebra
* slate: for parallel linear algebra (work in progress)
* spglib: for symmetries and k-point generation (work in progress)
* spdlog: Logging messages (different systems can have different log sinks)

INQ is work in progress, some of the features are not well tested or are not available at the moment.

## Components

* Electronic states, complex and real fields
* Solvers
* Diagonalization
* CPU/GPU generic algorithms
* Parallel distribution MPI/GPU/threads

## Basic installation

INQ has a few system dependencies and the environment ready in your system, for example in a desktop:
```bash
sudo apt install gfortran libblas-dev libboost-serialization-dev libboost-filesystem-dev libfftw3-dev libhdf5-dev liblapack-dev pkg-config python3-dev
```

Instructions for compiling

```bash 
git clone git@gitlab.com:npneq/inq.git --recursive
cd inq
mkdir build && cd build
# set up environment if necessary, e.g. export CUDACXX=/usr/local/cuda/bin/nvcc, or load modules
cmake .. --install-prefix=$HOME  # change prefix if necessary, e.g. $HOME/.local or /usr/local (needs root access)
cmake --build . --parallel
make install
make test
```

This instructions might be incomplete for your particular setup; 
see other possibilities in [`.gitlab-ci.yml`](https://gitlab.com/npneq/inq/blob/master/.gitlab-ci.yml).

## INQ papers

These papers describe the implementation of INQ. Please cite the relevant ones when you publish work that uses INQ.

- **INQ, a Modern GPU-Accelerated Computational Framework for (Time-Dependent) Density Functional Theory**.\
Xavier Andrade, Chaitanya Das Pemmaraju, Alexey Kartsev, Jun Xiao, Aaron Lindenberg, Sangeeta Rajpurohit, Liang Z. Tan, Tadashi Ogitsu, and Alfredo A. Correa\
[_J. Chem. Theory Comput._ 17, 7447–7467 (2021)](https://doi.org/10.1021/acs.jctc.1c00562) (open access)
- **Spin non-collinear real-time time-dependent density-functional theory and implementation in the modern GPU-accelerated INQ code**.\
Jacopo Simoni, Xavier Andrade, Wuzhang Fang, Andrew C. Grieder, Alfredo A. Correa, Tadashi Ogitsu, and Yuan Ping
[_APL Comput. Phys._, 1, 026108 (2025)](https://doi.org/10.1063/5.0290737)

## Introductory material

- (video) [INQ: a state-of-the art implementation of (TD)DFT for GPUs](https://www.youtube.com/watch?v=pM4wwYjb5Vo) at the MolSSI 2022 Workshop

- (video) [INQ: a state-of-the-art implemention of desnsity functional theory](https://www.youtube.com/watch?v=ufzOBKn9ocU&t=3195s) at 2021 USAfrI Workshop on Electronic Structure

## Release information 

INQ is licensed under the terms of the [Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/) (MPL).

``LLNL-CODE-803817``

CP Number: CP02279
Software Title: Electronic Structure Engine
Date Submitted: Wednesday, December 11, 2019
Date Accepted: Wednesday, January 29, 2020

The work is part of the activities of the Center for Non-Perturbative Studies of Functional Materials Under Non-Equilibrium Conditions (NPNEQ).

## Acknowledgment

This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, Materials Sciences and Engineering Division, under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.
