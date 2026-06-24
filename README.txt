Comparative Analysis of Machine Learning Classifiers for Early Detection of Breast and Prostate Cancer
======================================================================================================

About This Project
------------------

This repository contains all the code and data for my MSc dissertation project, which looks at how well four different machine learning models can detect breast cancer and prostate cancer. Instead of just training a model once and reporting results, I built a framework that runs 1024 different randomised experiments per model so the performance numbers are much more reliable and not just lucky splits.

On top of the classification work, I also wanted to understand which features actually matter most consistently across all those runs. So I built two different ranking methods (one based on the most frequently appearing features, and one based on clustering features by how similar they are) and then compared whether they agreed with each other. I also used Kernel PCA to check how much information you lose when you go from all features down to just the top 5, and I ran a full bootstrap evaluation to cross-check everything.

This is all written in Python using a Jupyter Notebook. If you want to see the results without running anything, just open the HTML file in your browser.


What Is In This Repository
---------------------------

README.txt                            -- this file
Cancer_Early_detection_Final.ipynb            -- the main notebook with all the analysis
Cancer_Early_detection_Final.html             -- the notebook already run, open in any browser
data/
    breast-cancer.csv                 -- Wisconsin Breast Cancer Dataset
    prostate_cancer_prediction.csv    -- Prostate Cancer Prediction Dataset
requirements.txt                      -- list of Python packages needed


The Datasets
------------

Breast Cancer (Wisconsin)
  - 569 patient records
  - 30 continuous features from biopsy images (radius, texture, concavity etc.)
  - Target: diagnosis column (B = Benign: 357 patients, M = Malignant: 212 patients)
  - Source: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

Prostate Cancer (Clinical)
  - 27,945 patient records
  - 29 features including PSA level, Gleason staging, BMI, age, DRE result etc.
  - Target: Biopsy_Result column (Benign: 19,549 patients, Malignant: 8,396 patients)
  - Source: https://www.kaggle.com/datasets/ankushpanday1/prostate-cancer-prediction-dataset

Both datasets are publicly available on Kaggle and do not contain any personally identifiable information.


How To Run It
-------------

You will need Python 3.8 or above. Install the required libraries with:

    pip install -r requirements.txt

Then launch Jupyter:

    jupyter notebook Cancer_Early_detection_Final.ipynb

Run all cells from top to bottom. The notebook loads the CSV files from the data/ folder so make sure those are in place first.

Fair warning: the 1024-run experiments take a while. On a normal laptop you are probably looking at somewhere between 20 and 40 minutes total, mostly because of the prostate cancer dataset which has nearly 28,000 rows. I put a subsample cap on the bootstrap section for prostate cancer to keep it reasonable.

If you just want to see what the results look like, open Cancer_Early_detection_Final.html in Chrome or Firefox and everything is already there.


Required Libraries
------------------

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
statsmodels
jupyter


What The Notebook Actually Does
--------------------------------

The notebook is split into two big sections: Section 2 for breast cancer and Section 3 for prostate cancer. Both follow exactly the same pipeline so the results are directly comparable.

Data cleaning and preparation
  I checked for nulls and duplicates (there were none in either dataset), detected outliers using the IQR method, and capped them using Winsorization rather than deleting rows. All features were kept for modelling since removing them based on VIF would have defeated the point of the feature importance analysis later.

Exploratory data analysis
  Basic statistics, class distribution charts, correlation heatmaps, and feature distribution plots to understand what we are working with before modelling.

The 1024-run experimental framework
  I wrote a custom randomisation function that creates a different train/test split for each seed from 1 to 1024. SMOTE is applied only to the training partition in each run to balance the classes without leaking anything into the test set. Two protocols are used: one on cancer-only patients (to find which features matter most within the positive class) and one on the full dataset (for overall model evaluation).

Feature importance and ranking
  After 1024 runs, each model has produced 1024 lists of the top 5 most important features. I then applied two different methods to rank these:

  MODE ranking: The feature that appears most frequently across all 1024 lists gets Rank 1. Then that feature is removed and we find the next most frequent one for Rank 2, and so on. The relative weights are normalised so they always add up to 100%.

  Dendrogram ranking: Features are clustered by how correlated they are using hierarchical clustering with Ward linkage. The ranking then picks the most representative feature from each cluster in turn, which helps avoid selecting five features that are all basically measuring the same thing.

  Both rankings are compared and any feature that appears in both lists is considered a strong predictor.

Kernel PCA and data loss
  I replaced standard PCA with Kernel PCA (using an RBF kernel) to better handle any non-linear structure in the data. The idea here is to measure how much variance you retain when you go from all features down to just the top 5. Separate plots show the cumulative variance for the full feature set, the MODE top 5, and the dendrogram top 5.

Covariance analysis
  A 5 by 5 covariance matrix is computed for the top 5 features from each ranking method, for each model, for both the cancer-only and full dataset versions. This tells you how strongly the top features are linearly related to each other, which matters clinically because if two features are very highly covariate, a clinician might be able to substitute one for the other when one is not available.

Model training and evaluation
  Each of the four models was evaluated across 1024 runs and results were averaged three ways: simple arithmetic mean, micro-averaging (pooling all confusion matrices), and a 95% confidence interval. Sensitivity was treated as the most important metric throughout because missing a cancer case is much worse than a false alarm.

Bootstrap evaluation
  On top of the 1024 custom-split runs, I ran a full OOB bootstrap evaluation with 1024 iterations. Each iteration samples the dataset with replacement, trains on the bootstrap sample, and tests on the rows that were not selected. The 95% confidence interval here uses the percentile method (2.5th to 97.5th) rather than assuming a normal distribution. Results are plotted side by side against the custom-split means so you can see how consistent the two approaches are.

The four models used
  Random Forest: 200 trees, square-root feature sampling, balanced class weights
  ANN (MLP): three hidden layers (128, 64, 32 neurons), ReLU, Adam optimiser, early stopping
  Decision Tree: max depth 10, balanced class weights, Gini impurity splits
  Gradient Boosting: sequential additive ensemble, balanced weights


Known Limitations
-----------------

The prostate cancer dataset is quite categorical and does not have many strong continuous discriminative features, which limits what any model can realistically achieve with it. This reflects the data rather than a problem with the methodology.

The Kernel PCA variance numbers for prostate cancer are approximations because I had to fit on a 2,000-row subsample to keep computation manageable.

The study does not include SHAP values or other explainability tools, which would be a useful next step for clinical interpretability.


References
----------

Agyemang et al. (2025). Addressing Class Imbalance Problem in Health Data Classification.
Applied Computational Intelligence and Soft Computing.
https://doi.org/10.1155/acis/1013769

Hicks et al. (2022). On evaluation metrics for medical applications of artificial intelligence.
Scientific Reports, 12(1).
https://doi.org/10.1038/s41598-022-09954-8

Jayaseeli et al. (2025). Multimodal feature-optimized approaches for cancer classification.
Scientific Reports, 15(1).
https://doi.org/10.1038/s41598-025-23195-5

Khalid et al. (2023). Breast Cancer Detection and Prevention Using Machine Learning.
Diagnostics, 13(19).
https://doi.org/10.3390/diagnostics13193113

Rainio, Teuho and Klen (2024). Evaluation metrics and statistical tests for machine learning.
Scientific Reports, 14(1).
https://doi.org/10.1038/s41598-024-56706-x

Salmi et al. (2024). Handling imbalanced medical datasets: review of a decade of research.
Artificial Intelligence Review, 57(10).
https://doi.org/10.1007/s10462-024-10884-2

Siddavatam and Shinde (2025). A Hybrid Literature Review on Handling Imbalanced Medical Data.
Expert Systems with Applications.
https://doi.org/10.1016/j.eswa.2025.129004


Data Attribution
----------------

Wisconsin Breast Cancer Dataset:
Yasserh (2022). Kaggle. https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

Prostate Cancer Prediction Dataset:
Ankushpanday1 (2025). Kaggle. https://www.kaggle.com/datasets/ankushpanday1/prostate-cancer-prediction-dataset


