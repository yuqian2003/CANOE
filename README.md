# Beyond Regularity: Modeling Chaotic Mobility Patterns for Next Location Prediction 🧠

CANOE is a novel model designed to predict a user's next destination by effectively balancing their periodic and chaotic movement patterns, while also leveraging contextual information for improved accuracy.


## Introduction 🌟

The analysis of human mobility is a pivotal task with broad applications, from optimizing urban resources to providing personalized navigation. A significant challenge in this field is accurately predicting a user's next location, a task complicated by the dual nature of human movement: it is both periodic (e.g., daily commutes) and chaotic (e.g., spontaneous trips). 

To address these limitations, Our project introduces an novel model that integrates a biologically inspired chaotic attention mechanism to better balance chaotic and periodic mobile pattern.

For more details, please refer to our paper: [CANOE](https://arxiv.org/abs/2509.11713v1)

## Key Modules ✨

- 🌐 Multimodal Contextual Embedding Module: Aiming at transforming heterogeneous ``who-when-where'' contexts into unified vector representations
- 📝 Tri-Pair Interaction Encoder: To capture pairwise relationships among user, time, and location modalities
- 🔄 Cross Context Attentive Decoder: Integrating the features of pairwise interaction by aligning ``who-when-where'' dependencies

## Framework 🏗️

<div align="center">
  <img src="Fig/CANOE.jpg" alt="CANOE Framework" width="800"/>
  <br>
  <em>Illustration of the CANOE framework.</em>
</div>

## Main Experimental Results 📊

Empirical experiments conducted on two real-world datasets demonstrate superior performance compared to existing SOTA baselines.

| Method | Acc@1 (traffic camera) | Acc@3 (traffic camera) | Acc@5 (traffic camera) | Acc@10 (traffic camera) | MRR (traffic camera) | Acc@1 (mobile phone) | Acc@3 (mobile phone) | Acc@5 (mobile phone) | Acc@10 (mobile phone) | MRR (mobile phone) |
|---|---|---|---|---|---|---|---|---|---|---|
| **1-MMC** | 23.61 | 39.50 | 44.43 | 48.29 | 32.42 | 29.48 | 45.68 | 49.54 | 52.46 | 38.21 |
| **Graph-Flashback** | 35.69 (0.03) | 55.64 (0.08) | 63.73 (0.05) | 72.25 (0.05) | 48.18 (0.03) | 37.61 (0.03) | 59.62 (0.02) | 65.86 (0.02) | 71.88 (0.03) | 50.31 (0.02) |
| **SNPM** | 36.43 (0.05) | 56.18 (0.02) | 63.74 (0.04) | 71.45 (0.02) | 48.58 (0.01) | 37.99 (0.07) | 59.89 (0.02) | 66.03 (0.04) | 71.92 (0.01) | 50.60 (0.04) |
| **DeepMove** | 35.89 (0.07) | 51.60 (0.07) | 57.72 (0.07) | 65.15 (0.04) | 46.08 (0.05) | 37.38 (0.04) | 56.84 (0.03) | 63.10 (0.02) | 69.88 (0.03) | 49.11 (0.02) |
| **Flashback** | 34.89 (0.06) | 54.92 (0.09) | 62.88 (0.04) | 71.00 (0.07) | 47.33 (0.04) | 37.39 (0.03) | 59.64 (0.05) | 65.96 (0.04) | 72.01 (0.06) | 50.22 (0.02) |
| **STAN** | 29.92 (0.10) | 49.70 (0.12) | 57.81 (0.08) | 66.24 (0.10) | 42.39 (0.08) | 36.40 (0.09) | 56.43 (0.10) | 62.15 (0.12) | 67.77 (0.12) | 48.06 (0.09) |
| **GETNext** | 35.53 (0.11) | 54.26 (0.14) | 61.27 (0.21) | 68.64 (0.12) | 47.15 (0.11) | 36.93 (0.09) | 59.44 (0.11) | 65.75 (0.40) | 71.80 (0.64) | 49.89 (0.09) |
| **Trans-Aux** | 36.69 (0.12) | 53.97 (0.13) | 60.38 (0.11) | 67.50 (0.07) | 47.52 (0.09) | 38.52 (0.31) | 56.66 (0.11) | 61.76 (0.18) | 67.28 (0.18) | 49.24 (0.12) |
| **CSLSL** | 36.96 (0.12) | 55.02 (0.13) | 61.67 (0.11) | 68.79 (0.07) | 48.17 (0.09) | 37.86 (0.16) | 60.22 (0.07) | 66.52 (0.02) | 71.94 (0.01) | 50.51 (0.11) |
| **MCLP-LSTM** | 39.90 (0.06) | 58.32 (0.07) | 65.14 (0.07) | 72.43 (0.07) | 51.28 (0.05) | 39.42 (0.16) | 60.74 (0.07) | 66.95 (0.06) | 72.98 (0.06) | 51.81 (0.08) |
| **MCLP-Attention** | 40.11 (0.05) | 58.44 (0.05) | 65.30 (0.04) | 72.58 (0.05) | 51.46 (0.02) | 39.65 (0.02) | 61.02 (0.05) | 67.18 (0.06) | 73.15 (0.05) | 52.04 (0.03) |
| **CANOE** | **45.37 (0.09)** | **64.43 (0.18)** | **71.03 (0.15)** | **77.78 (0.16)** | **56.86 (0.12)** | **40.92 (0.01)** | **63.04 (0.12)** | **69.41 (0.07)** | **75.49 (0.07)** | **53.69 (0.04)** |
| **Improvement (%)** | 13.11 | 10.25 | 8.78 | 7.16 | 10.49 | 3.20 | 3.30 | 3.32 | 3.20 | 3.17 |


## Performance Comparison Across Chaotic Levels 📊 
Below is the full evaluation table comparing CANOE and baselines across entropy thresholds on two datasets: TC and MP.


| Dataset       | Threshold | Model             | Acc@1         | Acc@3         | Acc@5         | Acc@10        | MRR           |
|---------------|-----------|-------------------|---------------|---------------|---------------|---------------|---------------|
| Traffic Camera| 0.75      | MCLP-LSTM         | 39.50 (0.06)  | 57.78 (0.16)  | 64.55 (0.01)  | 71.75 (0.02)  | 50.80 (0.01)  |
|               |           | MCLP-Attention    | 39.67 (0.06)  | 57.93 (0.06)  | 64.73 (0.02)  | 71.90 (0.06)  | 50.97 (0.02)  |
|               |           | CANOE(w/o CNOA)   | 43.80 (0.05)  | 63.12 (0.24)  | 70.07 (0.34)  | 77.27 (0.31)  | 55.58 (0.12)  |
|               |           | CANOE             | 45.00 (0.11)  | 64.10 (0.16)  | 70.77 (0.26)  | 77.58 (0.27)  | 56.54 (0.14)  |
|               | 0.80      | MCLP-LSTM         | 38.78 (0.07)  | 57.19 (0.16)  | 64.06 (0.01)  | 71.36 (0.02)  | 50.18 (0.01)  |
|               |           | MCLP-Attention    | 38.97 (0.05)  | 57.34 (0.08)  | 64.24 (0.02)  | 71.51 (0.06)  | 50.36 (0.02)  |
|               |           | CANOE(w/o CNOA)   | 43.12 (0.05)  | 62.61 (0.24)  | 69.67 (0.35)  | 76.97 (0.31)  | 55.02 (0.12)  |
|               |           | CANOE             | 44.38 (0.10)  | 63.63 (0.16)  | 70.40 (0.27)  | 77.30 (0.27)  | 56.03 (0.14)  |
|               | 0.85      | MCLP-LSTM         | 37.27 (0.07)  | 55.84 (0.15)  | 62.89 (0.01)  | 70.39 (0.01)  | 48.82 (0.01)  |
|               |           | MCLP-Attention    | 37.49 (0.04)  | 56.02 (0.06)  | 63.07 (0.02)  | 70.56 (0.05)  | 49.03 (0.01)  |
|               |           | CANOE(w/o CNOA)   | 41.72 (0.07)  | 61.43 (0.26)  | 68.69 (0.36)  | 76.23 (0.32)  | 53.81 (0.14)  |
|               |           | CANOE             | 43.09 (0.11)  | 62.55 (0.17)  | 69.51 (0.29)  | 76.64 (0.28)  | 54.92 (0.15)  |
|               | 0.90      | MCLP-LSTM         | 35.40 (0.12)  | 53.96 (0.16)  | 61.13 (0.03)  | 68.89 (0.00)  | 47.03 (0.03)  |
|               |           | MCLP-Attention    | 35.69 (0.03)  | 54.15 (0.06)  | 61.38 (0.02)  | 69.10 (0.05)  | 47.29 (0.00)  |
|               |           | CANOE(w/o CNOA)   | 40.04 (0.06)  | 59.79 (0.27)  | 67.22 (0.37)  | 75.03 (0.36)  | 52.25 (0.16)  |
|               |           | CANOE             | 41.63 (0.13)  | 61.17 (0.17)  | 68.27 (0.31)  | 75.65 (0.32)  | 53.57 (0.17)  |
| Mobile Phone  | 0.75      | MCLP-LSTM         | 37.01 (0.05)  | 59.08 (0.15)  | 65.31 (0.10)  | 71.41 (0.05)  | 49.77 (0.07)  |
|               |           | MCLP-Attention    | 37.34 (0.01)  | 59.33 (0.08)  | 65.70 (0.15)  | 71.92 (0.40)  | 50.09 (0.03)  |
|               |           | CANOE(w/o CNOA)   | 37.82 (0.06)  | 60.85 (0.00)  | 67.63 (0.02)  | 74.16 (0.03)  | 51.16 (0.04)  |
|               |           | CANOE             | 38.76 (0.04)  | 61.45 (0.08)  | 68.01 (0.09)  | 74.34 (0.08)  | 51.87 (0.05)  |
|               | 0.80      | MCLP-LSTM         | 36.20 (0.05)  | 58.39 (0.15)  | 64.70 (0.11)  | 70.87 (0.05)  | 49.04 (0.07)  |
|               |           | MCLP-Attention    | 36.56 (0.00)  | 58.66 (0.08)  | 65.09 (0.16)  | 71.38 (0.41)  | 49.39 (0.04)  |
|               |           | CANOE(w/o CNOA)   | 37.05 (0.06)  | 60.18 (0.01)  | 67.07 (0.02)  | 73.69 (0.03)  | 50.48 (0.05)  |
|               |           | CANOE             | 38.02 (0.08)  | 60.83 (0.06)  | 67.49 (0.10)  | 73.90 (0.05)  | 51.21 (0.05)  |
|               | 0.85      | MCLP-LSTM         | 35.17 (0.06)  | 57.41 (0.15)  | 63.82 (0.11)  | 70.09 (0.04)  | 48.07 (0.09)  |
|               |           | MCLP-Attention    | 35.56 (0.04)  | 57.71 (0.10)  | 64.25 (0.17)  | 70.60 (0.44)  | 48.44 (0.06)  |
|               |           | CANOE(w/o CNOA)   | 36.07 (0.10)  | 59.22 (0.01)  | 66.24 (0.03)  | 73.00 (0.03)  | 49.55 (0.06)  |
|               |           | CANOE             | 37.06 (0.09)  | 59.88 (0.04)  | 66.69 (0.07)  | 73.24 (0.04)  | 50.32 (0.05)  |
|               | 0.90      | MCLP-LSTM         | 34.32 (0.05)  | 56.27 (0.18)  | 62.70 (0.12)  | 69.05 (0.05)  | 47.11 (0.08)  |
|               |           | MCLP-Attention    | 34.78 (0.08)  | 56.62 (0.09)  | 63.15 (0.18)  | 69.57 (0.44)  | 47.54 (0.09)  |
|               |           | CANOE(w/o CNOA)   | 35.26 (0.12)  | 58.10 (0.08)  | 65.22 (0.01)  | 72.11 (0.00)  | 48.66 (0.09)  |
|               |           | CANOE             | 36.25 (0.05)  | 58.82 (0.10)  | 65.72 (0.09)  | 72.36 (0.07)  | 49.43 (0.05)  |




## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/yuqian2003/CANOE.git
cd CANOE
```

2. Install dependencies:
```bash
conda env create -f create_env.yml
conda activate canoe
```

3. Model Training
Run the experiments in Traffic Camera:
```bash
bash run_tc.sh
```
Run the experiments in Mobile Phone dataset:
```bash
bash run_mp.sh
```


## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@misc{wu2025regularitymodelingchaoticmobility,
      title={Beyond Regularity: Modeling Chaotic Mobility Patterns for Next Location Prediction}, 
      author={Yuqian Wu and Yuhong Peng and Jiapeng Yu and Xiangyu Liu and Zeting Yan and Kang Lin and Weifeng Su and Bingqing Qu and Raymond Lee and Dingqi Yang},
      year={2025},
      eprint={2509.11713},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.11713}, 
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
