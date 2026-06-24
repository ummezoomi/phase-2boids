# U.M.E.R Engine — Phase 2: High-Density Boids Showcase

![GPU Accelerated](https://img.shields.io/badge/Compute-GPU%20Accelerated-00E676?style=flat-square)
![Deterministic](https://img.shields.io/badge/Architecture-Sort--Free%20Deterministic-blue?style=flat-square)

> **Part of the U.M.E.R (Uniform Memory Encoded Representation) GPU Compute Core**

---

### Academic Preprint & Citation
This showcase acts as an empirical validation model for the architectural mechanics detailed in our manuscript submitted to the **Journal of Supercomputing**:

* **Title:** *U.M.E.R: A Deterministic Sort Free Architecture for High Density Physical & Agentic Systems*
* **Author:** Muhammad Umer
* **Preprint DOI:** [https://doi.org/10.21203/rs.3.rs-9848255/v1](https://doi.org/10.21203/rs.3.rs-9848255/v1)

---

## Overview

The **Phase 2 Showcase** demonstrates the scalable agentic navigation capabilities of the GPU-based U.M.E.R architecture. Using a classic 3D Boids (Flocking/Swarming) model, this repository tests a fundamental claim of the framework: **high-density spatial querying does not require memory sorting.**

In traditional GPU compute paradigms, simulating massive, dense autonomous agents requires expensive bounding volume hierarchies (BVH), k-d trees, or continuous Morton-code sorting to resolve neighbor lookups. This implementation bypasses tree-generation entirely.

## Architectural Innovations

* **Sort-Free Neighbor Resolution:** Agents are routed into fixed `8x8x8` TensorBlocks using deterministic spatial hashing. Because the spatial grid acts as an absolute physical ledger, an agent discovering its neighbors is reduced to a mathematically rigid, instant memory offset grab.
* **Zero Stochastic Drift:** Unlike standard probabilistic swarm approximations, every agent’s velocity, alignment, cohesion, and separation vector is resolved with 100% absolute determinism. 
* **L1/L2 Cache Locality:** By keeping agent data coupled strictly to its mapped spatial coordinate rather than an arbitrary thread index, the CUDA kernels achieve near-peak spatial locality, massively reducing VRAM bandwidth bottlenecks.

## Running the Showcase

The execution pipeline and renderer are encapsulated inside `phase-2boids.ipynb`.

### Prerequisites
* CUDA-capable GPU (NVIDIA HPC/RTX class recommended for peak scatter-gather throughput)
* Python 3.10+
* `cupy` / configured U.M.E.R GPU bindings

### Execution
Launch the notebook to compile the bare-metal kernels and initialize the swarm:
```bash
jupyter notebook phase-2boids.ipynb
