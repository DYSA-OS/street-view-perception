# Place Pulse 2.0: Walk Road Scoring(beautiful and clean)
**[Place Pulse 2.0](https://paperswithcode.com/dataset/place-pulse-2-0)** Introduced by Dubey et al. in "Deep Learning the City : Quantifying Urban Perception At A Global Scale"

> Place Pulse is a crowdsourcing effort that aims to map which areas of a city are perceived as safer, livelier, wealthier, more active, beautiful and friendly. By asking users to select images from a pair, Place Pulse collected more than 1.5 million reports that evaluate more than 100,000 images from 56 cities.

We extract street view of **walk road** 8,246 images from 40 cities in Place Pulse 2.0.
You can download images from this [link](https://drive.google.com/file/d/1goKUZP-0LDefLjUKMQAykpmheSofZBqI/view?usp=sharing).

The goal of this project is to analyze street views of roads, evaluate their **aesthetic appeal** and **cleanliness**, and mark areas that need improvement on a map.
To achieve this, we built and compared the performance of various models for predicting scores based on images.

1. **Baseline Model**: Predicts scores directly from the raw images.
2. **Semantic Segmentation-Based Model**: Segments objects in images and utilizes this information for prediction.
3. **Prompt-Based Model**: Generates textual descriptions of images as prompts for prediction.

The scores, originally ranging from 1 to 10, were transformed into three classification categories for training a **classification model**:

- **0 (Dissatisfied)**: Scores of 3 or below
- **1 (Neutral)**: Scores between 4 and 6
- **2 (Satisfied)**: Scores of 7 or above

However, the models did not learn effectively and failed to produce meaningful results. 
Suggesting the potential integration of natural language models into existing semantic segmentation-based research.

## Results
|  model   | class      | accuracy | f1_score |
|:--------:|:----------:|:--------:|:--------:|
| baseline | beautiful  | 0.5267   | 0.4524   |
| baseline |   clean    | 0.6218   | 0.5038   |
| segment  | beautiful  | 0.4867   | 0.4691   |
| segment  |   clean    | 0.5764   | 0.5061   |
| prompt   | beautiful  | 0.5576   | 0.3992   |
| prompt   |   clean    | 0.6339   | 0.4919   |

## Data(# of image)
|train|validation|test|
|:--:|:--:|:--:|
|5772|824|1650|

### Segmentation
HRNet(Semantic Segmentation): https://github.com/CSAILVision/semantic-segmentation-pytorch

<p align="center">
  <img src="https://github.com/user-attachments/assets/c0c6ebde-65d9-46ce-a3fa-0802f2ba60c1" width=70%>
</p>

### Prompt
LLaVA(Image &rarr; Prompt): https://github.com/camenduru/LLaVA

![Screenshot 2025-03-15 at 8 12 37 PM](https://github.com/user-attachments/assets/cae79474-7640-4dbb-bec2-e74322245424)

## Model
### 1. Baseline: DenseNet121
```python baseline.py```

### 2. Segmentation: XGBoost
```python segment.py```

### 3. Prompt: XLNet
```python prompt.py```
