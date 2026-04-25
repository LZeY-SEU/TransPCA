# TransPCA

Here provides the codes to reproduce the numerical experiments in the paper "Knowledge Transfer across Multiple Principal Component Analysis Studies".

Some experiments are based on larger scale public datasets, which can be found at https://www.kaggle.com/datasets/jindongwang92/crossposition-activity-recognition (Wang, et al. 2018) and https://www.sciencedirect.com/science/article/abs/pii/S016794731630041X (Gross and
Tibshirani, 2016), respectively.

## Summary

We present a two-step transfer learning algorithm that utilizes multiple source PCA studies to enhance the estimation accuracy of the target PCA task. The first step integrates shared subspace information using a Grassmannian barycenter method. The shared subspace estimator is then used to estimate the target private subspace in the second step.

## Citation

```
@article{li2024knowledge,
  title={Knowledge transfer across multiple principal component analysis studies},
  author={Li, Zeyu and Qin, Kangxiang and He, Yong and Zhou, Wang and Zhang, Xinsheng},
  journal={arXiv preprint arXiv:2403.07431},
  year={2024}
}
```
