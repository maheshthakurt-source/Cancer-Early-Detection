# Cancer-Early-Detection

About This Project 
------------------ 
This repository contains all the code and data for my MSc dissertation project, which looks at how well four different machine learning models can detect breast cancer and prostate cancer. Instead of just training a model once and reporting results, I built a framework that runs 1024 different randomised experiments per model so the performance numbers are much more reliable and not just lucky splits. On top of the classification work, I also wanted to understand which features actually matter most consistently across all those runs. So I built two different ranking methods (one based on the most frequently appearing features, and one based on clustering features by how similar they are) and then compared whether they agreed with each other. I also used Kernel PCA to check how much information you lose when you go from all features down to just the top 5, and I ran a full bootstrap evaluation to cross-check everything. This is all written in Python using a Jupyter Notebook. If you want to see the results without running anything, just open the HTML file in your browser. 

What Is In This Repository 
--------------------------- 
README.txt 
-- this file Cancer Early detection.ipynb 
-- the main notebook with all the analysis Cancer Early detection.html 
-- the notebook already run, open in any browser breast-cancer.csv 
-- Wisconsin Breast Cancer Dataset prostate_cancer_prediction.csv 
-- Prostate Cancer Prediction Dataset requirements.txt 
-- list of Python packages needed 

**Dataset Links** 
--------------------- 
https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset 
https://www.kaggle.com/datasets/ankushpanday1/prostate-cancer-prediction-dataset
