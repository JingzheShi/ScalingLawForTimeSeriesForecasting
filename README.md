# ScalingLawForTimeSeriesForecasting
The repository for experiment codes for our work *Scaling Law for Time Series Forecasting*. **Coming soon.**

**Scaling Law for Time Series Forecasting**

**Authors:** [Jingzhe Shi](mailto:shi-jz21@mails.tsinghua.edu.cn)$^{1}$, [Qinwei Ma](mailto:mqw21@mails.tsinghua.edu.cn)$^{1}$, [Huan Ma](mailto:mah21@mails.tsinghua.edu.cn), [Lei Li](mailto:lilei@di.ku.dk).

&emsp; $^1$: equal contribution

**Abstract:** Scaling law that rewards large dataset, complex model and enhanced data granularity has been observed in various fields in deep learning. Yet, previous works for time series forecasting have cast doubt on scaling behaviors of deep learning methods for time series forecasting: more capable models do not always outperform less capable models and longer input horizon sometimes hurt performance.
  We propose a theory for scaling law for time series forecasting that can explain these seemingly abnormal behaviors. 
  We take into account the impact of dataset size and model size, as well as time series data granularity, especially horizon, that previous theories do not lay emphasize on.
  %Our theory not only explains scaling behaviors from a dataset size and model width perspective (like previously proposed theories), but we  with a focus on the impact of time series data granularity (especially on horizon) [and other dimension: dataset size, model width KUOCHENGLIANGJU]. 
  Furthermore, we conduct experiments with different models on a variety of time series forecasting datasets, which (1) verify the validity of scaling law on dataset size and model size in the area of time series forecasting and (2) validate our proposed theory, especially about the impact of horizon: for a certain model and a certain amount of training data, there exists an optimal horizon which increases with the expansion of available training data. 
  %Our theory can explain the previously proved successful design components including patches, low-pass-filter, etc. 
  We hope our findings may inspire new models targeting specific time series forecasting datasets of limited size, as well as large foundational datasets and models for time series forecassting in future works.

This repository contains no new models or datasets. Instead, it contains codes like training scripts, and PCA analysis tools for time series that are used in our work.

We acknowledge the code bases provided by several previously proposed fantastic works, including [TimeSeriesLibrary](https://github.com/thuml/Time-Series-Library), [iTransformer](https://github.com/thuml/iTransformer), [ModernTCN](https://github.com/luodhhh/ModernTCN).
