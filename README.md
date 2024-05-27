# Scaling Law For Time Series Forecasting

Repository for the [paper](https://arxiv.org/abs/2405.15124):

**Scaling Law for Time Series Forecasting**

**Authors:** [Jingzhe Shi](mailto:shi-jz21@mails.tsinghua.edu.cn)$^\star$, [Qinwei Ma](mailto:mqw21@mails.tsinghua.edu.cn)$^\star$, [Huan Ma](mailto:mah21@mails.tsinghua.edu.cn), [Lei Li](mailto:lilei@di.ku.dk).

&emsp; $^\star$: equal contribution

**Abstract:** Scaling law that rewards large datasets, complex models and enhanced data granularity has been observed in various fields of deep learning. Yet, studies on time series forecasting have cast doubt on scaling behaviors of deep learning methods for time series forecasting: while more training data improves performance, more capable models do not always outperform less capable models, and longer input horizons may hurt performance for some models.
  We propose a theory for scaling law for time series forecasting that can explain these seemingly abnormal behaviors. 
  We take into account the impact of dataset size and model complexity, as well as time series data granularity, particularly focusing on the look-back horizon, an aspect that has been unexplored in previous theories.
  Furthermore, we empirically evaluate various models using a diverse set of time series forecasting datasets, which (1) verifies the validity of scaling law on dataset size and model complexity within the realm of time series forecasting, and (2) validates our theoretical framework, particularly regarding the influence of look back horizon. We hope our findings may inspire new models targeting time series forecasting datasets of limited size, as well as large foundational datasets and models for time series forecasting in future works.

This repository contains no new models or datasets. Instead, it contains codes like training scripts, and PCA analysis tools for time series that are used in our work.

We acknowledge the code bases provided by several previously proposed fantastic works, including [TimeSeriesLibrary](https://github.com/thuml/Time-Series-Library), [iTransformer](https://github.com/thuml/iTransformer), [ModernTCN](https://github.com/luodhhh/ModernTCN).
