# A Hybrid Learning Framework for Adaptive Decision-Making in SDN-Based Healthcare IoT Systems

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the official implementation of the paper:

> **"A Hybrid LSTM-Reinforcement Learning Framework for Adaptive Decision-Making in SDN-Based Healthcare IoT Systems"**

The framework integrates a supervised LSTM predictor with deep reinforcement learning (DQN and PPO) to jointly optimize network resource allocation and health-event prioritization in SDN-enabled IoMT networks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [1. Train the LSTM Predictor](#1-train-the-lstm-predictor)
  - [2. Train DRL Agents](#2-train-drl-agents)
  - [3. Evaluate on Test Data](#3-evaluate-on-test-data)
  - [4. Reproduce Figures and Tables](#4-reproduce-figures-and-tables)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project presents a hybrid framework where:

- **LSTM Predictor** (Layer 1): Forecasts critical health events from the last five vital-sign readings (8 features).
- **DRL Agent** (Layer 2): Uses the prediction alongside raw vital signs, battery level, and network congestion to select one of four discrete network actions:
  - `0`: Normal Route
  - `1`: High Priority Alert
  - `2`: Reduce Sampling Rate
  - `3`: Reroute via Backup

The framework is evaluated on a real-world IoMT dataset (5,095 records, patient-wise split: 70% training, 15% validation, 15% testing). Two DRL algorithms are compared:
- **Enhanced DQN** (with dueling architecture, double Q-learning, prioritized replay, and critical-event oversampling)
- **PPO** (with weighted entropy for critical states)

Baselines: **Rule‑Based** and **Model Predictive Control (MPC)**.

---

## Key Features

- ✅ Hybrid LSTM-DRL architecture for proactive health monitoring.
- ✅ Discrete action space (4 actions) for SDN network control.
- ✅ Composite reward function balancing medical urgency, alignment, network performance, and battery.
- ✅ 10-seed training for statistical robustness.
- ✅ Prioritized replay with critical-event oversampling (DQN).
- ✅ Weighted entropy for critical states (PPO).
- ✅ Full evaluation pipeline with 12 performance metrics.

---

## Repository Structure
