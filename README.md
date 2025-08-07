# CNOLP: Contextual Next Location Prediction

This is the PyTorch implementation of the CNOLP and helps readers to reproduce the results in the paper "**Chaotic Neural Oscillator Network for Next Location Prediction**".

## Model Framework
<p align="middle" width="100%">
  <img src="fig/framework.png" width="60%"/>
</p>

## Installation

### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/yuqian2003/CNOLP.git
cd CNOLP

# Create conda environment
conda env create -f create_env.yaml
conda activate cnolp
```

### Option 2: Using Pip
```bash
# Clone the repository
git clone https://github.com/yuqian2003/CNOLP.git
cd CNOLP

# Install dependencies
pip install -r requirements.txt
```

## Configurations
For both datasets, the embedding dimensions of the proposed model are set to 16.  
The Transformer encoder consists of 2 layers, each with 4 attention heads and a dropout rate of 0.1.  
The Arrival Time Estimator has 4 attention heads.  
We train MCLP for 50 epochs with a batch size of 256. 

## Hyperparameters
All hyperparameter settings are saved in the `.yml` files under the respective dataset folder under `saved_models/`. 

For example, `saved_models/TC/settings.yml` contains hyperparameter settings of MCLP for Traffic Camera Dataset. 

## Usage

### Data Preparation
1. Unzip `data/TC.zip` to `data/TC`. The two files are training data and testing data.
2. Unzip `data/MP.zip` to `data/MP` for Mobile Phone Dataset.

### Running the Model

#### Traffic Camera Dataset (TC):
```bash
# For MCLP model:
python ./model/run.py --dataset TC --dim 16 --topic 400 --at attn

# For MCLP(LSTM) model:
python ./model/run.py --dataset TC --dim 16 --topic 400 --at attn --encoder lstm

# With custom oscillator type and bandwidth:
python ./model/run.py --dataset TC --dim 16 --topic 400 --at attn --type 2 --bandwidth 1.5
```

#### Mobile Phone Dataset (MP):
```bash
# For MCLP model:
python ./model/run.py --dataset MP --dim 16 --topic 400 --at attn

# For MCLP(LSTM) model:
python ./model/run.py --dataset MP --dim 16 --topic 400 --at attn --encoder lstm
```

### Parameters
- `--dataset`: Dataset name (TC or MP)
- `--dim`: Embedding dimension (default: 16)
- `--topic`: Number of LDA topics (default: 400)
- `--at`: Arrival time module type (attn or none)
- `--encoder`: Encoder type (trans or lstm)
- `--type`: Oscillator type (1-6, default: 1)
- `--bandwidth`: Bandwidth parameter for embedding (default: None)

## Citation
```
@inproceedings{xxxxx,
  title={xxxxxx},
  author={xxxxxx},
  booktitle={xxxxxx},
  pages={xxxxx},
  year={xxxxx}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
- GitHub: [@yuqian2003](https://github.com/yuqian2003)
