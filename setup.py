import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

setup(
    name = 'confidence_correctness_matrix',
    version = '1.0.1',
    author = 'Jesús S. Aguilar-Ruiz, Alejandro García Conde',
    #author_email=,
    description = 'Confidence-Correctness Matrix',
    long_description = (HERE / "README.md").read_text(encoding='utf-8'), 
    long_description_content_type = "text/markdown",
    #url=,
    packages=["confidence_correctness_matrix"],
    # classifiers=,
    # python_requires=,
    install_requires = ["numpy","matplotlib"],
    license = "BSD 3-Clause License"
)
