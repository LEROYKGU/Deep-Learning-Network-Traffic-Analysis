# Overview

This project presents a large-scale deep learning pipeline for network traffic analysis and prediction using sequential packet-level data.

The objective is to model temporal traffic behavior and predict:

* Inter-arrival packet timing (T)
* Packet length (pkt_len)

The project compares multiple neural network architectures combining:

* GRU recurrent layers
* Temporal convolution (Conv1D)
* Hybrid sequence modeling
The study was conducted on a real-world dataset containing more than 2.4 million network traffic windows, making this project representative of industrial-scale sequential learning problems.
Data URL :
´´´bash  https://www.nature.com/articles/s41597-025-04876-2?utm_source=perplexity
´´´
---
# Project Objectives

The main objectives of this project are:

* Model sequential network traffic behavior
* Predict future packet characteristics
* Compare recurrent and hybrid deep learning architectures
* Optimize performance-complexity tradeoffs
* Analyze training stability in large-scale DL systems
---
# Problem Statement

Modern network infrastructures generate massive sequential traffic data. Predicting future packet behavior can improve:

* Traffic optimization
* Congestion management
* Intrusion detection preprocessing
* QoS prediction
* Adaptive routing systems

This project formulates the problem as a multi-output regression task:

* Predict packet inter-arrival time (T)
* Predict packet length (pkt_len)

using temporal packet history windows.
---
# Dataset Description

The dataset contains approximately:

* 2,408,457 rows
* 12 original features
* Sequential network packet observations

Input windows were generated using a sliding-window approach:

* Sequence length: 50 timesteps
* Final input shape: (50, 5)
---
# Selected Features

After correlation analysis and feature selection, the following features were retained:
```bash
[
    'local_rate',
    'ip_len',
    'local_density',
    'burst_position',
    'silence_before'
]
```
These features capture:

* Local traffic intensity
* Packet density
* Burst dynamics
* Silence periods
* Packet structural behavior
---
# Data Preparation Pipeline

The preprocessing pipeline includes:
## Data Normalization
* Min-Max normalization per session
* float16 optimization for memory reduction
## Sliding Window Construction
For each packet:

* Input: previous 50 packets
* Output: next packet targets `[T, pkt_len]`
## Dataset Splitting
- Training: 72.3%
- Validation: 12.8%
- Test: 15.0%
---
# Deep Learning Architectures

The project compares three architectures.
---
# Model 1 — Double GRU Baseline

Architecture:
```bash
GRU(128) → GRU(32)
```
Characteristics:

* Pure recurrent architecture
* Two stacked GRU layers
* Baseline temporal modeling approach
## Results
| Metric             | Value    |
| ------------------ | -------- |
| Test Loss          | 0.100108 |
| Test MAE (T)       | 0.006283 |
| Test MAE (pkt_len) | 0.093823 |


Parameters:

- 68,594 trainable parameters
---
# Model 2 — Conv1D + GRU Hybrid

Architecture:
```bash
Conv1D(128) → GRU(64)
```
This hybrid model introduces:

+ Temporal convolution for local pattern extraction
+ GRU for long-range dependencies

Advantages:

+ Faster training
+ Better generalization
+ Lower parameter count
## Results
| Metric             | Value    |
| ------------------ | -------- |
| Test Loss          | 0.077774 |
| Test MAE (T)       | 0.003645 |
| Test MAE (pkt_len) | 0.074128 |

Parameters:

- 43,986 parameters

This model achieved:

* Best performance
* Lowest complexity
* Fastest convergence
---
# Model 3 — Conv1D + Double GRU

Architecture:
```bash
Conv1D(128) → GRU(64) → GRU(32)
```
This architecture introduces:

+ Hierarchical temporal representation
+ Multi-level recurrent processing
## Results
| Metric             | Value    |
| ------------------ | -------- |
| Test Loss          | 0.099418 |
| Test MAE (T)       | 0.005937 |
| Test MAE (pkt_len) | 0.093477 |


Parameters:

- 52,498 parameters

Despite increased complexity, the model underperformed due to training instability.
---
# Global Performance Comparison
| Model           | Parameters | Test Loss  | MAE(T)      | MAE(pkt_len) |
| --------------- | ---------- | ---------- | ----------- | ------------ |
| GRU ×2          | 68,594     | 0.1001     | 0.00628     | 0.09382      |
| Conv1D + GRU    | 43,986     | **0.0778** | **0.00365** | **0.07413**  |
| Conv1D + GRU ×2 | 52,498     | 0.0994     | 0.00594     | 0.09348      |

The hybrid Conv1D + GRU model demonstrated the best tradeoff between:

* Accuracy
* Complexity
* Training speed
---
# Key Engineering Insights
## Strengths
* Large-scale dataset
* Structured ML pipeline
* Efficient memory optimization using float16
* Rigorous architectural comparison
* Multi-output prediction design
## Identified Limitations
* Learning rate instability (lr=0.01)
* Potential temporal leakage
* Limited training epochs
* Absence of naive baseline comparison
---
# Recommendations

Future improvements include:

* Reduce learning rate to 0.001
* Increase training epochs
* Session-level train/test split
* Attention mechanisms
* Float32 precision benchmarking 
* Transformer-based architectures
---
# Technologies Used
## Programming Language
* Python 3.13.5
## Deep Learning Framework
* TensorFlow / Keras
## Libraries
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
---
# Repository Structure
```bash
Deep-Learning-Network-Traffic-Analysis/
│
├── analysis/
|   └── traffic_analysis.ipynb
|   └── pcap_feature_extractor.py 
|
├── reports/
│   └── rapport_comparatif_Traffic_Analysis.docx
│
└── README.md
```
---
# Installation

Clone the repository:
```bash
git clone https://github.com/LEROYKGU/your-repository-name.git
```
Navigate to the project directory:
```bash
cd your-repository-name
```
Install dependencies:
```bash
pip install -r requirements.txt
```
---
# Usage

Launch Jupyter Notebook:
```bash
jupyter notebook
```
Run notebooks sequentially:

1. Data preprocessing
2. Feature engineering
3. Model training
4. Evaluation and comparison
---
# Applications

This project can be applied to:

* Intelligent traffic prediction
* Network optimization
* Telecom analytics
* Cybersecurity preprocessing
* Time-series forecasting
* Large-scale sequential modeling
---
# Learning Outcomes

This project demonstrates competencies in:

* Deep Learning
* Sequential modeling
* Network traffic analytics
* GRU architectures
* Conv1D temporal feature extraction
* Large-scale ML engineering
* Experimental model comparison
* Performance optimization
---
# Conclusion

The study demonstrates that a lightweight hybrid architecture combining:
```bash
Conv1D + GRU
```
significantly outperforms deeper recurrent architectures for this traffic prediction task.

The results also highlight an important deep learning principle:

`Increased architectural complexity does not guarantee improved performance without stable optimization dynamics.`
---
# Author

KGU
Data Scientist | Deep Learning Practitioner | AI Engineer
```bash
GitHub: https://github.com/LEROYKGU
```
# License

This project is licensed under the MIT License.
