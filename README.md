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

Empirical experiments conducted on two real-world datasets demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/yuqian2003/CNOLP.git
cd CNOLP
```

2. Install dependencies:
```bash
conda env create -f create_env.yml
conda activate cnolp
```

3. Run the experiments in Traffic Camera or Mobile Phone dataset:
```bash
bash run_tc.sh
```
```bash
bash run_mp.sh
```


## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@article{xxx,
  title={CNOLP: Chaotic Neural Oscillator Network for Next Location Prediction},
  author={Wu, Yuqian and Peng, Yuhong and Yu, Jiapeng and Liu, Xiangyu and Yan, Zeting and Lin, Kang and Su, Weifeng and Qu, Bingqing and Raymond S.T. Lee and Yang, Dingqi },
  journal={xxxxxxxxxx},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
