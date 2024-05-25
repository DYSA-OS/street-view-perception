# Place Pulse 2.0
**[Place Pulse 2.0](https://paperswithcode.com/dataset/place-pulse-2-0)** Introduced by Dubey et al. in "Deep Learning the City : Quantifying Urban Perception At A Global Scale"

> Place Pulse is a crowdsourcing effort that aims to map which areas of a city are perceived as safer, livelier, wealthier, more active, beautiful and friendly. By asking users to select images from a pair, Place Pulse collected more than 1.5 million reports that evaluate more than 100,000 images from 56 cities.

## Result
|  model   | class_name | accuracy | f1_score |
|:--------:|:----------:|:--------:|:--------:|
| baseline | beautiful  | 0.557576 | 0.399198 |
| baseline |   clean    | 0.633939 | 0.491914 |
| segment  | beautiful  | 0.486667 | 0.469052 |
| segment  |   clean    | 0.576364 | 0.506088 |
|   llm    | beautiful  | 0.557576 | 0.399198 |
|   llm    |   clean    | 0.633939 | 0.491914 |

## Dataset
```python dataset.py```

- Train set size: 5772
- Validation set size: 824
- Test set size: 1650

## Model
### 1. Classification
DenseNet121: https://github.com/liuzhuang13/DenseNet

#### step
```python baseline.py```

### 2. Segmentation
HRNet: https://github.com/CSAILVision/semantic-segmentation-pytorch

#### step
- ```git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git```
- ```cd semantic-segmentation-pytorch```
- ```DOWNLOAD_ONLY=1 ./demo_test.sh```
- ```python converter.py```
- ```cd ..```
- ```python segment.py```

### 3. LLM
- LLaVA: https://github.com/camenduru/LLaVA
- XLNet: https://github.com/zihangdai/xlnet

#### step
- ```git clone -b v1.0 https://github.com/camenduru/LLaVA```
- ```cd LLaVA```
- ```python img2prompt.py```
- ```cd ..```
- ```python llm.py```
