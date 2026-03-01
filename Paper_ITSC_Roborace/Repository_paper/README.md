# Trajectory Planning and Control at the Limits:  
## An Open Experimental Benchmark on the RoboRacer Platform

This repository accompanies the paper:

**“Trajectory Planning and Control at the Limits: an Open Experimental Benchmark on the RoboRacer Platform”**

It provides the core control modules and the MS-NN training implementations used in the experimental evaluation presented in the paper.

The repository is intended to support transparency and reproducibility of the reported results.  
It does not include the full ROS2 stack, hardware configuration, or deployment scripts.

---

## 1. Overview

Autonomous racing represents a challenging benchmark for high-performance motion planning and control. The proposed framework combines:

- Geometric steering controllers  
- A Model-Structured Neural Network (MS-NN) for steering correction  
- A Forward–Backward velocity replanning module (FBGA)  

The hybrid architecture integrates physics-informed neural modeling with real-time velocity replanning to improve trajectory tracking and lap-time performance at the handling limits.

All experiments were conducted on a 1:10 scale F1Tenth autonomous racing platform using ROS2 and onboard NVIDIA Jetson hardware.

---

## 2. Repository Content

The repository contains standalone implementations of the four steering controllers used in the experimental comparison, as well as the MS-NN training files evaluated in the paper.

---

### Baseline Controllers

- `pp_steering_node`  
  Pure Pursuit steering controller used as geometric baseline.

- `clothoid_steering_node`  
  Clothoid-based steering controller used as geometric baseline.

These controllers are included for benchmarking purposes and do not represent novel contributions of the paper.

---

### Proposed Hybrid Architectures

- `pp_nn_fbga_steering_node`  
  Pure Pursuit + MS-NN steering correction + optional FBGA velocity replanning.

- `clothoid_nn_fbga_steering_node`  
  Clothoid steering + MS-NN correction + optional FBGA velocity replanning.

In both hybrid configurations:

- The MS-NN augments the geometric steering command.
- The FBGA module performs real-time velocity replanning based on GGV constraints.
- The velocity replanner can be enabled or disabled via internal flags.

---

## 3. MS-NN Training Files

The repository includes the Python training files used to construct and train the Model-Structured Neural Networks (MS-NN) evaluated in the paper.

Two different neural architectures are provided:

- `control_steer_lateral_dynamics_base`  
  Baseline MS-NN architecture corresponding to the version originally introduced by Da Lio.

- `control_steer_lateral_dynamics_extended`  
  Extended MS-NN architecture proposed in this work to improve steering performance.

These two networks are directly compared in the experimental analysis presented in the paper.

Both files:

- Are implemented using *nnodely*  
- Explicitly define the structured neural architecture  
- Include the complete training procedure  

This allows full reproducibility of the neural model formulation, including the structured dynamics representation and the training workflow adopted in the experimental study.

---

## 4. Experimental Deployment

The controllers provided in this repository were executed:

- As ROS2 nodes  
- On an NVIDIA Jetson onboard computer  
- On a F1Tenth autonomous racing platform  

Mapping, raceline generation, LiDAR processing, and full ROS2 launch configurations are not included in this repository.

The focus of this release is the core control logic and the MS-NN training implementations used in the paper.

---

## 5. Reproducibility Statement

This repository provides:

- The exact control implementations used in the experimental benchmark  
- The complete MS-NN formulation and training files (base and extended)  
- The hybrid controller configurations evaluated in the paper  

The purpose is to enable independent inspection of the algorithms, the neural model structure, and the experimental methodology.
