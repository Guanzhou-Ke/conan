# CONAN: Contrastive Fusion Networks for Multi-view Clustering
The official repos of "CONAN: Contrastive Fusion Networks for Multi-view Clustering".

- Submitted at: [IEEE BigData 2021](https://bigdataieee.org/BigData2021/index.html)

- Status: Accepted (short paper).

# Abstract

With the development of big data, deep learning has made remarkable progress on multi-view clustering. Multi-view fusion is a crucial technique for the model obtaining a common representation. However, existing literature adopts shallow fusion strategies, such as weighted-sum fusion and concatenating fusion, which fail to capture complex information from multiple views. In this paper, we propose a novel fusion technique, entitled contrastive fusion, which can extract consistent representations from multiple views and maintain the characteristic of view-specific representations. Specifically, we study multi-view alignment from an information bottleneck perspective and introduce an intermediate variable to align each view-specific representation. Furthermore, we leverage a single-view clustering method as a predictive task to ensure the contrastive fusion is working. We integrate all components into a unified framework called CONtrAstive fusion Network (CONAN). Experiment results on five multi-view datasets demonstrate that CONAN outperforms state-of-the-art methods. 

# Architecture

![arch](./imgs/architecture.png)

# Environment

- Python 3.8
- PyTorch 1.8.0
- CUDA 10.2


# Training

All our experiments are put in `./experiments`, data files under `data/processed`.

Note: Before you run the program firstly, you should run `datatool/load_dataset` to generate dataset.

You could quickly run our experiments by: `python train.py -c [config name]`.

For example: `python train.py -mnist`


# Fusion Results

![vis](./imgs/vis.png)

# Citation

```
@inproceedings{CONAN,
  author    = {Guanzhou Ke and
               Zhiyong Hong and
               Zhiqiang Zeng and
               Zeyi Liu and
               Yangjie Sun and
               Yannan Xie},
  title     = {{CONAN:} Contrastive Fusion Networks for Multi-view Clustering},
  booktitle = {2021 {IEEE} International Conference on Big Data (Big Data), Orlando,
               FL, USA, December 15-18, 2021},
  pages     = {653--660},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/BigData52589.2021.9671851},
  doi       = {10.1109/BigData52589.2021.9671851},
}
```
