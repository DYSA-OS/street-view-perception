# Place Pulse 2.0
**[Place Pulse 2.0](https://paperswithcode.com/dataset/place-pulse-2-0)** Introduced by Dubey et al. in "Deep Learning the City : Quantifying Urban Perception At A Global Scale"

> Place Pulse is a crowdsourcing effort that aims to map which areas of a city are perceived as safer, livelier, wealthier, more active, beautiful and friendly. By asking users to select images from a pair, Place Pulse collected more than 1.5 million reports that evaluate more than 100,000 images from 56 cities.

## Dataset
```python dataset.py```

## Model
### 1. Classification
DenseNet121: https://github.com/liuzhuang13/DenseNet

```python baseline.py```

### 2. Segmentation
HRNet: https://github.com/CSAILVision/semantic-segmentation-pytorch

- ```git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git```
- ```cd semantic-segmentation-pytorch```
- ```sh demo_test.sh```

### 3. LLM
- LLaVA: https://github.com/camenduru/LLaVA
- XLNet: https://github.com/zihangdai/xlnet