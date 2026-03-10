# Trajectory Planning and Control at the Limits:  
## An Open Experimental Benchmark on the RoboRacer Platform

This repository accompanies the paper:

**“Trajectory Planning and Control at the Limits: an Open Experimental Benchmark on the RoboRacer Platform”**

It provides the core control modules, the MS-NN training implementations, the raceline generation scripts, and the experimental datasets used in the evaluation presented in the paper.

The repository is intended to support transparency and reproducibility of the reported results.  
It does not include the full ROS2 stack, hardware configuration, or deployment scripts.


---

## 1. Overview

Autonomous racing represents a challenging benchmark for high-performance motion planning and control. The proposed framework combines:

- Geometric steering controllers  
- A Model-Structured Neural Network (MS-NN) for steering correction  
- A Forward–Backward velocity replanning module (FBGA)  

The hybrid architecture integrates physics-informed neural modeling with real-time velocity replanning to improve trajectory tracking and lap-time performance at the handling limits.

All experiments were conducted on the RoboRacer 1:10 scale autonomous racing platform using ROS2 and onboard NVIDIA Jetson hardware.


---

## 2. Repository Content

The repository provides the code and datasets used in the experimental study presented in the paper.

It includes:

- implementations of the steering controllers evaluated in the benchmark
- training scripts for the Model-Structured Neural Networks (MS-NN)
- scripts for offline raceline generation
- experimental telemetry datasets collected during multiple testing campaigns
- Python dependency files required to reproduce the MS-NN training environment

The main directories of the repository are:

- `controllers/` – steering controllers used in the experiments
- `Raceline_generation/` – scripts for offline raceline computation
- `telemetries/` – datasets collected during the experimental campaigns
- `msnn_training/` – MS-NN training scripts
- `requirements.txt` – Python dependencies for the training environment


---

## 3. Steering Controllers

The repository includes standalone implementations of the steering controllers used in the experimental benchmark.

Two baseline geometric controllers are provided:

- `pp_steering_node`  
  Pure Pursuit steering controller.

- `clothoid_steering_node`  
  Clothoid-based steering controller.

These controllers serve as geometric baselines for the experimental comparison.

In addition, two hybrid controllers integrating the proposed architecture are provided:

- `pp_nn_fbga_steering_node`  
  Pure Pursuit controller augmented with MS-NN steering correction and optional FBGA velocity replanning.

- `clothoid_nn_fbga_steering_node`  
  Clothoid-based controller augmented with MS-NN steering correction and optional FBGA velocity replanning.

In the hybrid architectures:

- the MS-NN augments the geometric steering command
- the FBGA module performs forward–backward velocity replanning
- the velocity replanner can be enabled or disabled via internal flags


---

## 4. MS-NN Training Files
The repository includes the Python training files used to construct and train the Model-Structured Neural Networks (MS-NN) evaluated in the paper, including both the baseline architecture introduced by Da Lio et al. and the extended formulation proposed in this work.

Two different neural architectures are provided:

-  `control_steer_lateral_dynamics_base`  
  Baseline MS-NN architecture corresponding to the model-structured neural network introduced by Da Lio et al. in the paper:
  
  *A Mental Simulation Approach for Learning Neural-Network Predictive Control (in Self-Driving Cars)*  
  
  Available on IEEE Xplore: https://ieeexplore.ieee.org/document/9234399

  This neural architecture represents the inverse model of the vehicle lateral dynamics and serves as the baseline structured neural formulation adopted in this work.

- `control_steer_lateral_dynamics_extended`  
  Extended MS-NN architecture proposed in this work to improve steering performance.

These two networks are directly compared in the experimental analysis presented in the paper.

Both files:

- Are implemented using *nnodely*  
- Explicitly define the structured neural architecture  
- Include the complete training procedure  

The MS-NN architectures are implemented using **nnodely**, an open-source Python framework designed for the development of Model-Structured Neural Networks (MS-NNs). The framework enables the construction of neural models that integrate prior knowledge from physics, control theory, and system dynamics directly into the neural network architecture. This hybrid approach combines the learning capabilities of neural networks with structured model-based formulations, enabling improved interpretability, reduced training data requirements, and reliable deployment in real-world control applications.

nnodely GitHub repository: https://github.com/tonegas/nnodely

This allows full reproducibility of the neural model formulation, including the structured dynamics representation and the training workflow adopted in the experimental study.


---

## 5. Raceline Generation

The `Raceline_generation` directory contains the scripts used to compute the reference racelines adopted in the experimental setup.

For each track, a time-optimal raceline is generated following the approach proposed by Christ et al. in the paper:

*Time-optimal trajectory planning for a race car considering variable tyre-road friction coefficients*  
*Vehicle System Dynamics*

Available on Taylor & Francis Online: https://doi.org/10.1080/00423114.2019.1704804

This method formulates a point-mass optimal control problem constrained by the vehicle **g–g–v diagram**, enabling the computation of a time-optimal trajectory along the track.

The raceline generation is performed **offline** starting from a track map representation. The resulting raceline is then used as the reference trajectory for the steering controllers during the experiments.

The provided scripts allow users to inspect and reproduce the raceline computation procedure adopted in the experimental benchmark.


---

## 6. Telemetry Datasets

The `telemetries` directory contains experimental datasets collected during multiple testing campaigns with the RoboRacer platform.

These datasets include the telemetry used for:

- MS-NN training
- controller evaluation
- experimental analysis reported in the paper

The datasets are organized according to the controller configuration used during the experiment.

These datasets enable inspection of the experimental telemetry used in the paper, supporting transparency and reproducibility of the reported results.

### File Naming Convention

Telemetry files follow a naming convention that encodes the controller configuration used during the experiment.

General format:

[Controller Type]_MSNN_FBGA_LAxxxx

Where:

- `PP` indicates the **Pure Pursuit steering controller** type
- `CL` indicates the **Clothoid-based steering controller** type
- `MSNN` indicates that the **Model-Structured Neural Network augmentation** was active
- `FBGA` indicates that the **Feed-Back Gain Adaptation** mechanism was enabled
- `laxxxx` identifies the **lookahead parameter range**

Not all tags are necessarily present in every filename.  
For example, a dataset may correspond to a **baseline controller**, a controller augmented with **MSNN**, with **FBGA**, or with both.

The lookahead (la) parameter is implemented as a **dynamic anchor**, meaning that the preview distance varies during driving as a function of vehicle speed and track curvature.

The digits following `la` encode the **minimum and maximum lookahead distances** used by the controller:

- the **first two digits** represent the **minimum lookahead distance**
- the **last two digits** represent the **maximum lookahead distance**

Distances are expressed in **meters**, with an **implicit decimal point after the first digit**.

For example:

`la1117`

indicates a lookahead distance dynamically varying between **1.1 m and 1.7 m** along the track.


---

## 7. Python Environment and Dependencies

The project relies on the nnodely framework for implementing Model-Structured Neural Networks (MS-NNs).
All required Python dependencies are listed in the requirements.txt file located in the root directory of the repository.

### Environment Setup

We recommend creating a dedicated Python environment before installing the dependencies.

Using **conda**:

```bash
conda create -n <env_name> python=3.10   # replace <env_name> with the desired name
conda activate <env_name>
```

Install the required Python packages for this project:

```bash
pip install -r requirements.txt
```

### Installing nnodely

The nnodely package can be installed directly from PyPI:

```bash
pip install nnodely
```

Alternatively, it can be installed from source:

```bash
git clone https://github.com/tonegas/nnodely.git
cd nnodely
pip install -r requirements.txt
pip install . 
```


---

## 8. Experimental Deployment

The controllers provided in this repository were executed:

- As ROS2 nodes  
- On an NVIDIA Jetson onboard computer  
- On the RoboRacer 1:10 autonomous racing platform  

Mapping, raceline generation, LiDAR processing, and full ROS2 launch configurations are not included in this repository.

The focus of this release is the core control logic and the MS-NN training implementations used in the paper.

---

## 9. Reproducibility Statement

This repository provides:

- The exact control implementations used in the experimental benchmark  
- The complete MS-NN formulation and training files (base and extended)  
- The hybrid controller configurations evaluated in the paper  

The purpose is to enable independent inspection of the algorithms, the neural model structure, and the experimental methodology.