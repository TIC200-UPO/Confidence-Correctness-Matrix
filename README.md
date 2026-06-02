# Confidence-Correctness Matrix

The package provides the methods for building the confidence-correctness matrix, for its visualization in a horizontal bar chart and for computing different probabilistic metrics. The confidence-correctness matrix is an innovative method to understand the behavior of a prediction model for classification problems.

This matrix provides information about the degree of confidence that the classifier has in its own predictions, indicating whether it is robust and reliable or uncertain and doubtful. This method has two variants: the class-independent confidence-correctness matrix and the class-specific confidence-correctness matrix depending on the kind of analysis required.

By analyzing the data provided by them, our goal is to improve the reliability and explainability of prediction models and to provide users with a clearer understanding of why a model has a high or low confidence about its predictions.

## Installation

Serendipity Matrix can be installed from [PyPI](https://pypi.org/project/confidence_correctness_matrix/)

```bash
pip install confidence_correctness_matrix
```

Or you can clone the repository and run:

```bash
pip install .
```

## Sample usage

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from ucimlrepo import fetch_ucirepo

iris  = fetch_ucirepo(id=53) 
X, y = iris.data.features, iris.data.targets.squeeze()

model = GaussianNB().fit(X, y)
y_score = model.predict_proba(X)

prob_conf_matrix = prob_confusion_matrix(y, y_score, labels=classes)
prob_acc = prob_accuracy_score(y, y_score)
prob_f1 = prob_f1_score(y, y_score, average="weighted")

H, L = confidence_matrices(y, y_score)
lambda_H, lambda_L = confidence_weights(y, y_score)

ci_confCorrM = confidence_correctness_matrix(y, y_score, class_specific=False)
cd_confCorrM = confidence_correctness_matrix(y, y_score, class_specific=True)

plot_confidence(y, y_score)
```

## Result sample

### Probabilistic confusion matrix
| Iris-setosa  |  Iris-versicolor | Iris-virginica |
|:------------:|:----------------:|:--------------:|
|      50      |         0        |        0       |
|      0       |       46.06      |       3.94     |
|      0       |        3.93      |      46.07     |

### High-confidence matrix (H)

| Iris-setosa  |  Iris-versicolor | Iris-virginica |
|:------------:|:----------------:|:--------------:|
|      50      |         0        |         0      |
|       0      |       45.374     |       2.314    |
|       0      |        2.644     |      45.715    |

lambda_H = 0.97365

### Low-confidence matrix (L)

| Iris-setosa  |  Iris-versicolor | Iris-virginica |
|:------------:|:----------------:|:--------------:|
|      0       |          0       |         0      |
|      0       |        0.686     |       1.626    |
|      0       |        1.285     |       0.356    |

lambda_L = 0.02635

### Confident-Correctness matrix

|                             | High-confidence | Low-confidence |  
|:---------------------------:|:---------------:|:--------------:|
|  Prob. mass on True class   |      0.941      |      0.007     |
| Prob. mass on Other classes |      0.033      |      0.019     |

### Class-specific Confidence-Correctness Matrix horizontal bar chart
[Class-specific Confidence-Correctness matrix](resources/example_confidence_correctness_iris.png)

For more detailed examples, please refer to the following [link](https://tic200-upo.github.io/Confidence-Correctness-Matrix/example.html).

## Citation

The methodology is described in detail in:

[1] J. S. Aguilar-Ruiz and A. García Conde, “”<!-- , Scientific Reports, 14:10759, 2024, doi: 10.1038/s41598-024-61365-z. Also, the mathematical background of the multiclass classification performance can be found in: in IEEE Access.-->

## Documentation

Full documentation is available [here](https://tic200-upo.github.io/Confidence-Correctness-Matrix/).
