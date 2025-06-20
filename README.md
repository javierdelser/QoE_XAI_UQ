
# Benchmarking Machine Learning Models for QoE Estimation in Video Streaming

## Project Description
This repository contains the code for the paper 'Benchmarking Machine Learning Models for QoE Estimation in Video Streaming'. The paper provides a comprehensive evaluation of various machine learning models for estimating Quality of Experience (QoE) in video streaming scenarios.

## Dataset
The dataset used in this project is SNESet, which includes various features related to video streaming and user experience. The dataset is available at [SNESet](https://github.com/YananLi18/SNESet).

## Machine Learning Models
The following machine learning models were evaluated in the paper:
- Linear Regression (Elastic Nets)
- Support Vector Machines
- Random Forests
- Neural Networks
- LightGBM
- XGBoost
- CatBoost
- Kolmogorov-Arnold Networks
- Ensemble Deep Random Vector Functional Link Networks
- GP-based Symbolic Regression

## XAI/UQ Techniques
The paper also explores Explainable AI (XAI) and Uncertainty Quantification (UQ) techniques to provide insights into model predictions and their reliability. 

## Results

![MAE and SMAPE](https://github.com/javierdelser/QoE_XAI_UQ/blob/main/img/MAE_SMAPE.png)

![Training and inference times](https://github.com/javierdelser/QoE_XAI_UQ/blob/main/img/training_inference_times.png)

![Confidence estimation via CP](https://github.com/javierdelser/QoE_XAI_UQ/blob/main/img/confidence_edrvfl.png)

![SHAP Waterfall plot](https://github.com/javierdelser/QoE_XAI_UQ/blob/main/img/SHAP_edRVFL_waterfalls.png)

## Citation
If you use this code in your research, please cite the paper as follows:
```
@article{bilbao2025benchmarking,
  title={Benchmarking Machine Learning Models for QoE Estimation in Video Streaming: Accuracy, Efficiency, Confidence and Explainability},
  author={Miren Nekane Bilbao, Mikel Getino-Petit, Javier Del Ser},
  journal={under review for its presentation in the 12th International Conference of Networks, Games, Control and Optimization (NETGCOOP)},
  year={2025}
}
```
