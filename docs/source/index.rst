Confident-Correctness
========

Welcome to Confident-Correctness's documentation!

The package provides the methods to provide the confidence-correctness matrix and for its visualization in a horizontal bar chart. The confidence-correctness matrix is an innovative method to understand the behavior of a prediction model for classification problems.

This matrix provides information about the degree of confidence that the classifier has in its own predictions, indicating whether it is robust and reliable or uncertain and doubtful. This method has two variants: the class-independent confidence-correctness matrix and the class-specific confidence-correctness matrix depending on the kind of analysis required.

By analyzing the data provided by them, our goal is to improve the reliability and explainability of prediction models and to provide users with a clearer understanding of why a model has a high or low confidence about its predictions.

Installation
============
Confidence-Correctness Matrix can be installed from `PyPI <https://pypi.org/project/confidence_correctness_matrix/>`_::

    pip install confidence_correctness_matrix

Or you can clone the repository and run:

    pip install .

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Examples of usage <./example.rst>
   Code documentation <./documentation.rst>