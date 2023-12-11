Code implementing FRESH (Jain et al. 2020, https://arxiv.org/abs/2005.00115). Note that the code is not vectorized.

Data: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

BertExtractors	Classes that implement the Top-k and Contiguous-k extractors for FRESH.

EDA Exploratory data analysis
Extractor Extracts the rationales from text
Predictor	Predicts the final output using the rationale
ResultsAnalysis	Computes accuracy scores and other analyses
SHAPExploration	Creates SHAP explanations