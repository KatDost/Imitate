# IMITATE
Machine Learning typically assumes that training and test set are independently drawn from the same distribution, but this assumption is often violated in practice which creates a bias. Many attempts to identify and mitigate this bias have been proposed, but they usually rely on ground-truth information. But what if the researcher is not even aware of the bias?

In contrast to prior work, our ICDM paper (Citation below) introduces a new method, IMITATE, to identify and mitigate Selection Bias in the case that we may not know if (and where) a bias is present, and hence no ground-truth information is available.

IMITATE investigates the dataset's probability density, then adds generated points in order to smooth out the density and have it resemble a Gaussian, the most common density occurring in real-world applications. If the artificial points focus on certain areas and are not widespread, this could indicate a Selection Bias where these areas are underrepresented in the sample.

We demonstrate the effectiveness of the proposed method in both, synthetic and real-world datasets. We also point out limitations and future research directions. The results are given in our ICDM paper, however, we offer a more detailed extended version of the paper here.

# Citation
If you want to use this implementation or cite Imitate in your publication, please cite the following ICDM paper:
```
Katharina Dost, Katerina Taskova, Patricia Riddle, and Jörg Wicker.
"Your Best Guess When You Know Nothing: Identification and Mitigation of Selection Bias."
In: 2020 IEEE International Conference on Data Mining (ICDM), IEEE, Forthcoming.
```

Bibtex:
```
@inproceedings{dost2020your,
title = {Your Best Guess When You Know Nothing: Identification and Mitigation of Selection Bias},
author = {Katharina Dost and Katerina Taskova and Pat Riddle and Jörg Wicker},
year = {2020},
date = {2020-11-17},
booktitle = {2020 IEEE International Conference on Data Mining (ICDM)},
publisher = {IEEE},
pubstate = {forthcoming}
}
```

# How to use IMITATE
Examples on how to use IMITATE (as well as all experiments and tests mentioned in the paper) are given in form of Jupyter Notebooks in the folder Experiments_Examples_Tests. In order to re-run them, they need to be placed together with the .py files. 

The implementation is structured as follows:
- IMITATE.py: Core algorithm incl. point generation (Paper Alg. 1,2; Sec. IV.D)
- Transformations.py: Outlier removal and transformation (Sec. IV.A)
- DensityEstimators.py: KDE and Hhistograms (Sec. IV.B)
- Distributions.py: Distribution fitting (Sec. IV.C)
- Confidence.py: Confidence estimation (Sec. IV.E)

Additional files for data and data handling are provided here:
- Datasets.py: Synthetic data generation
- Bias.py: Synthetic biases and data handling (split into train/test etc.)
