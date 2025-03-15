# Place Pulse 2.0: Walk Road Scoring(beautiful and clean)
**[Place Pulse 2.0](https://paperswithcode.com/dataset/place-pulse-2-0)** Introduced by Dubey et al. in "Deep Learning the City : Quantifying Urban Perception At A Global Scale"

> Place Pulse is a crowdsourcing effort that aims to map which areas of a city are perceived as safer, livelier, wealthier, more active, beautiful and friendly. By asking users to select images from a pair, Place Pulse collected more than 1.5 million reports that evaluate more than 100,000 images from 56 cities.

From Place Pulse 2.0, we extracted only images corresponding to **pedestrian roads**, resulting in a total of **8,246 images from 40 cities**.
You can download images from this [link](https://drive.google.com/file/d/1goKUZP-0LDefLjUKMQAykpmheSofZBqI/view?usp=sharing).

The goal of this project is to analyze street views of roads, evaluate their **aesthetic appeal** and **cleanliness**, and mark areas that need improvement on a map.
To achieve this, we built and compared the performance of various models for predicting scores based on images.

1. **Baseline Model**: Predicts scores directly from the raw images.
2. **Semantic Segmentation-Based Model**: Segments objects in images and utilizes this information for prediction.
3. **Prompt-Based Model**: Generates textual descriptions of images as prompts for prediction.

The scores, originally ranging from 1 to 10, were transformed into three classification categories for training a classification model:

- **0 (Dissatisfied)**: between 1 and 4
- **1 (Neutral)**: between 4 and 7
- **2 (Satisfied)**: between 7 and 10

## Results
The majority of the scores were distributed between 4 and 6, leading the model to predominantly predict 'neutral (1)'.
As a result, the accuracy was high, but the model was not useful in practice.

|  model   | class      | accuracy | f1_score | 
|:--------:|:----------:|:--------:|:--------:|
| baseline | beautiful  | 0.8170   | 0.7347   |
| baseline |   clean    | 0.8267   | 0.7510   |
| segment  | beautiful  | 0.8024   | 0.7320   |
| segment  |   clean    | 0.8200   | 0.7508   |
| prompt   | beautiful  | 0.8170   | 0.7347   |
| prompt   |   clean    | 0.8297   | 0.7525   |

### Confusion Matrix
| model    | beautiful  | clean     | 
|----------|--------------------------------|--------------------------------|
| baseline | `[[0, 260, 0],`<br>`[0, 1348, 0],`<br>`[0, 42, 0]]` | `[[0, 61, 0],`<br>`[0, 1364, 5],`<br>`[0, 220, 0]]` |
| segment  | `[[4, 255, 1],`<br>`[27, 1320, 1],`<br>`[1, 41, 0]]` | `[[0, 61, 0],`<br>`[1, 1350, 18],`<br>`[0, 217, 3]]` |
| prompt  | `[[0, 260, 0],`<br>`[0, 1348, 0],`<br>`[0, 42, 0]]` | `[[0, 61, 0],`<br>`[0, 1369, 0],`<br>`[0, 220, 0]]` |

## Data
### # of Image
|train|validation|test|
|:--:|:--:|:--:|
|5772|824|1650|

### Segment
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

### 2. Segment: HRNet + XGBoost
```python segment.py```

### 3. Prompt: LLaVA + XLNet
```python prompt.py```
