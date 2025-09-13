# Chaotic Neural Oscillator Network for Next Location Prediction 🧠

CNOLP is a novel model designed to predict a user's next destination by effectively balancing their periodic and chaotic movement patterns, while also leveraging contextual information for improved accuracy.


## Introduction 🌟

The analysis of human mobility is a pivotal task with broad applications, from optimizing urban resources to providing personalized navigation. A significant challenge in this field is accurately predicting a user's next location, a task complicated by the dual nature of human movement: it is both periodic (e.g., daily commutes) and chaotic (e.g., spontaneous trips). 

To address these limitations, Our project introduces an novel model that integrates a biologically inspired chaotic attention mechanism to better balance chaotic and periodic mobile pattern.


## Key Modules ✨

- 🌐 Multimodal Contextual Embedding Module: Aiming at transforming heterogeneous ``who-when-where'' contexts into unified vector representations
- 📝 Tri-Pair Interaction Encoder: To capture pairwise relationships among user, time, and location modalities
- 🔄 Cross Context Attentive Decoder: Integrating the features of pairwise interaction by aligning ``who-when-where'' dependencies

## Framework 🏗️

<div align="center">
  <img src="Fig/CNOLP.jpg" alt="CNOLP Framework" width="800"/>
  <br>
  <em>Illustration of the CNOLP framework.</em>
</div>

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/AgenticMemory.git
cd AgenticMemory
```

2. Install dependencies:
Option 1: Using venv (Python virtual environment)
```bash
# Create and activate virtual environment
python -m venv a-mem
source a-mem/bin/activate  # Linux/Mac
a-mem\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda
```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

3. Run the experiments in LoCoMo dataset:
```python
python test_advanced.py 
```

**Note:** To achieve the optimal performance reported in our paper, please adjust the hyperparameter k value accordingly. 

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@article{xu2025mem,
  title={A-mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
