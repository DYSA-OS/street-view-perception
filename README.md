# Place Pulse 2.0: Walk Road Scoring(beautiful and clean)
**[Place Pulse 2.0](https://paperswithcode.com/dataset/place-pulse-2-0)** Introduced by Dubey et al. in "Deep Learning the City : Quantifying Urban Perception At A Global Scale"

> Place Pulse is a crowdsourcing effort that aims to map which areas of a city are perceived as safer, livelier, wealthier, more active, beautiful and friendly. By asking users to select images from a pair, Place Pulse collected more than 1.5 million reports that evaluate more than 100,000 images from 56 cities.

We extract street view of **walk road** 8,246 images from 40 cities.
You can download images from this [link](https://drive.google.com/file/d/1goKUZP-0LDefLjUKMQAykpmheSofZBqI/view?usp=sharing).

## Results
|  model   | class      | accuracy | f1_score |
|:--------:|:----------:|:--------:|:--------:|
| baseline | beautiful  | 0.5267   | 0.4524   |
| baseline |   clean    | 0.6218   | 0.5038   |
| segment  | beautiful  | 0.4867   | 0.4691   |
| segment  |   clean    | 0.5764   | 0.5061   |
| prompt   | beautiful  | 0.5576   | 0.3992   |
| prompt   |   clean    | 0.6339   | 0.4919   |

### Confusion Matrix
|Class|Baseline|Segment|Prompt|
|:--:|:--:|:--:|:--:|
|beautiful|$$\begin{matrix} 780 & 140 & 0 \\ 585 & 89 & 0 \\ 51 & 5 & 0 \end{matrix}$$|$$\begin{matrix} 590 & 328 & 2 \\ 459 & 213 & 2 \\ 36 & 20 & 0 \end{matrix}$$|$$\begin{matrix} 920 & 0 & 0 \\ 674 & 0 & 0 \\ 56 & 0 & 0 \end{matrix}$$|
|clean|$$\begin{matrix} 18 & 377 & 1 \\ 38 & 1008 & 0 \\ 3 & 205 & 0 \end{matrix}$$|$$\begin{matrix} 51 & 337 & 8 \\ 124 & 898 & 24 \\ 29 & 177 & 2 \end{matrix}$$|$$\begin{matrix} 0 & 396 & 0 \\ 0 & 1046 & 0 \\ 0 & 208 & 0 \end{matrix}$$|

## Data(# of image)
- Train: 5772
- Validation: 824
- Test: 1650

## Model
### 1. Baseline
- DenseNet121: https://github.com/liuzhuang13/DenseNet
- ```python baseline.py```

### 2. Segmentation
- HRNet(Semantic Segmentation): https://github.com/CSAILVision/semantic-segmentation-pytorch
- ```python segment.py```

### 3. Prompt
- LLaVA(Image &rarr; Prompt): https://github.com/camenduru/LLaVA
- XLNet: https://github.com/zihangdai/xlnet
- ```python prompt.py```
