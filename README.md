# Confidence-Correctness Matrix

The package provides the methods to provide the confidence-correctness matrix and for its visualization in a horizontal bar chart. The confidence-correctness matrix is an innovative method to understand the behavior of a prediction model for classification problems.

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

# Loads the dataset
iris  = fetch_ucirepo(id=53) 
X, y = iris.data.features, iris.data.targets.squeeze()

classes = np.unique(y)
print(f"Classes: {classes}\n")

np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})

# Training and predict
model = GaussianNB().fit(X, y)
result = model.predict_proba(X)

# Calculates the probabilistic confusion matrix and the probabilistic metrics
prob_conf_matrix = prob_confusion_matrix(y, result, labels=classes)
prob_acc = prob_accuracy_score(y, result)
prob_b_acc = prob_balanced_accuracy_score(y, result)
prob_prec = prob_precision_score(y, result, average="micro")
prob_recall = prob_recall_score(y, result, average="macro")
prob_f1 = prob_f1_score(y, result, average="weighted")
prob_cohen_kappa = prob_cohen_kappa_score(y, result)
prob_m_corrcoef = prob_matthews_corrcoef(y, result)

print(f"Probabilistic confusion matrix:\n {prob_conf_matrix}\n")
print(f"Acc* = {prob_acc:.3f}, B_acc* = {prob_b_acc:.3f}, Prec* = {prob_prec:.3f}, MCC* = {prob_m_corrcoef:.3f}, Recall* = {prob_recall:.3f}, F1* = {prob_f1:.3f}, Cohen Kappa* = {prob_cohen_kappa:.3f}\n")

# Calculates the high-confidence and low-confidence matrices and their lambda values
H, L = confidence_matrices(y, result)
lambda_H, lambda_L = confidence_weights(y, result)

print(f"High-Confidence matrix:\n {H}")
print(f"lambda_H = {lambda_H:.3f}\n")
print(f"Low-Confidence matrix:\n {L}")
print(f"lambda_L = {lambda_L:.3f}\n")

# Calculates the class-independent Confidence-Correctnes matrix
ci_confCorrM = confidence_correctness_matrix(y, result, class_dependent=False)
print("Class-independent Confidence-Correctnes matrix values:")
for key, value in ci_confCorrM.items():
    print(f"{key}: {value:.3f}")

# Calculates the class-dependent Confidence-Correctnes matrix
cd_confCorrM = confidence_correctness_matrix(y, result, class_dependent=True)
print("\nClass-dependent Confidence-Correctnes matrix values:")
for key, value in cd_confCorrM.items():
    print(f"Class {key}")
    for key2, value2 in value.items():
        print(f"{key2}: {value2:.3f}")

# Plots the class-dependent Confidence-Correctnes matrix
plot_confidence(y, result)
```

## Result sample

### Probabilistic confusion matrix
| Iris-setosa  |  Iris-versicolor | Iris-virginica |
|:------------:|:----------------:|:--------------:|
|      50      |         0        |        0       |
|      0       |       46.06      |       3.94     |
|      0       |        3.93      |      46.07     |

Acc* = 0.94754

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

<!--![Class-specific serendipity matrix](Resources/Example_class-specific_serendipity_matrix_for_wine_dataset.png)-->

## Citation

The methodology is described in detail in:

[1] J. S. Aguilar-Ruiz and A. García Conde, “”<!-- , Scientific Reports, 14:10759, 2024, doi: 10.1038/s41598-024-61365-z. Also, the mathematical background of the multiclass classification performance can be found in: in IEEE Access.-->

## Documentation

Full documentation is available [here](https://tic200-upo.github.io/Confidence-Correctness-Matrix/).
