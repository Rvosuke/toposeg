# TopoSeg: Topology-Aware Nuclear Instance Segmentation

This repository contains a PyTorch implementation of the paper "TopoSeg: Topology-Aware Nuclear Instance Segmentation" by Hongliang He, Jun Wang, Pengxu Wei, Fan Xu, Xiangyang Ji, Chang Liu, Jie Chen. The paper presents a novel approach for nuclear instance segmentation that incorporates topological information to improve the accuracy and rationality of the segmentation results.

## Introduction

Nuclear instance segmentation is a critical task in pathology image analysis, enabling the quantitative assessment of nuclear morphology for cancer diagnosis and prognosis. However, accurate segmentation of individual nuclei remains challenging due to their dense distribution, blurred boundaries, and large variability in size, shape, and texture.

TopoSeg addresses these challenges by introducing a topology-aware segmentation framework. It extends the conventional pixel-wise optimization paradigm by incorporating meaningful topological constraints. The key components of TopoSeg include:

- **Topology Encoding**: A method to quantitatively represent the topological characteristics of nuclear instances in three-class probability maps (inside, boundary, background).
- **Topology-Aware Module (TAM)**: A module that encodes the persistence of topological structures in probability maps and introduces a topology-aware loss function to guide the model learning.
- **Adaptive Topology-Aware Selection (ATS)**: A strategy to efficiently focus on regions with high topological errors, enabling the model to handle dense and small nuclear instances.

## Installation

Clone the repository:
   ```
   git clone https://github.com/your_username/TopoSeg.git
   cd TopoSeg
   ```

## Model Architecture

The TopoSeg model consists of two main branches:
- **Semantic Branch**: A U-Net-like architecture that predicts the semantic categories of nuclei (e.g., carcinoma, epithelial, etc.).
- **Three-Class Branch**: A U-Net-like architecture that predicts the structural categories (inside, boundary, background) of nuclei.

The Topology-Aware Module (TAM) is applied to the output of the three-class branch to capture and optimize the topological structures. The Adaptive Topology-Aware Selection (ATS) module is used to efficiently select regions with high topological errors for further refinement.

## Results

TopoSeg achieves state-of-the-art performance on three nuclear instance segmentation datasets:
- MoNuSeg: PQ 0.625, AJI 0.643
- CPM17: PQ 0.705, AJI 0.756
- PBNuclei: PQ 0.528, AJI 0.509

Qualitative results demonstrate the effectiveness of TopoSeg in accurately segmenting nuclear instances, especially in challenging scenarios with dense and overlapping nuclei.

## Citation

If you find this work useful in your research, please consider citing the paper:

```bibtex
@inproceedings{he2023toposeg,
  title={TopoSeg: Topology-Aware Nuclear Instance Segmentation},
  author={He, Hongliang and Wang, Jun and Wei, Pengxu and Xu, Fan and Ji, Xiangyang and Liu, Chang and Chen, Jie},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
