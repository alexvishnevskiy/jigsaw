# Jigsaw
This repository contains code for pre-training and fine-tuning different nlp models for [jigsaw](https://www.kaggle.com/c/jigsaw-toxic-severity-rating) competiton.

## Code structure
**jigsaw/configs** - different configs for rnn, linear, cnn, roberta models.

**jigsaw/datasets** - scripts for datasets and dataloaders.

**jigsaw/models** - scripts for models classes and lightning modules.

**jigsaw/scripts** - scripts for training and inference.

**jigsaw/utils** - useful utils for loading vectors, sampling, callbacks, cleaning.
## Results
Private leaderboard: 278/2328

**CV:**

|                  | Accuracy |
|------------------|----------|
| lstm/glove       |  0.7166  |
| lstm/fasttext    |  0.7365  |
| linear/glove     |  0.7080  |
| linear/fasttext  |  0.7110  |
| cnn/glove        |  0.6732  |
| roberta-base     |  0.7506  |


**LB:**

|              | Public LB  | Private LB |
|--------------|------------|------------|
| roberta-base | 0.7969     | 0.7951     |
| ensemble     | 0.7475     | 0.7770     |


