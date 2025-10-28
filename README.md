## Acknowledgements

This repository provides the source code and training scripts for 3D medical image segmentation models that incorporate **location context**, as described in:

**“LocBAM: Advancing 3D Patch-Based Image Segmentation by Integrating Location Context”**  
Donnate Bridget Hooft*, Stefan M. Fischer*, Cosmin Bercea, Jan C. Peeken, Julia A. Schnabel  
(*Shared first authorship*)  
Technical University of Munich, Helmholtz Munich, and Munich Center of Machine Learning (MCML)  

[Conference/Journal details to be added]  
A full citation (BibTeX/APA) is provided in the `CITATION.cff` file.

---

## Repository Overview

This repository implements several strategies for integrating spatial and anatomical priors into patch-based 3D medical image segmentation models:

- **Coordinate-based models**
  - CoordConv layers for explicit coordinate encoding.
  - UNETR-style positional embeddings for global spatial representation.

- **Body Part Regression (BPR)**
  - Integration of normalized anatomical scores (pelvis–head range) as spatial priors for CT-based segmentation.

- **LocBAM (Location-Based Attention Module)**
  - A lightweight, axis-wise attention mechanism for incorporating anatomical awareness into convolutional encoders.
  - Extends the hierarchical attention concept from HANet to 3D volumetric data.
  - Provides robust, memory-efficient training under limited patch-to-volume coverage (PtVC).

---

## Framework and Implementation

- Implemented entirely with **PyTorch** and **MONAI**.
- All code was written from scratch; no proprietary or third-party segmentation components were reused.
- Training and evaluation follow the **nnU-Net** configuration for standardized benchmarking.
- Compatible with **BTCV**, **AMOS22**, and **KiTS23** datasets.

---

## Inspiration

The LocBAM architecture was conceptually inspired by the height-aware attention mechanism introduced in:

HANet: *Height-Aware Attention Networks for Semantic Segmentation*  
GitHub: [https://github.com/lhc1224/HANet](https://github.com/lhc1224/HANet)  
License: Creative Commons BY-NC 4.0

---

## License

This project is released under the **Apache License 2.0**, allowing free use, modification, and distribution with attribution.  
See the `LICENSE` file for details.

