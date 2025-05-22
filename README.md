#### Acknowledgements
Herein are the source codes and training scripts for 3D medical image segmentation models with location context, as presented in:

<!-- “Spatial Context within 3D Patch-based Medical Image Segmentation: Integrating Global and Relative Positional Priors”
Donnate Bridget Hooft, Stefan Fischer, et al.
Presented at [insert conference/journal if known]
(A citation in BibTeX/APA will be provided in the CITATION.cff file.) -->

This repository contains implementations for several architectures incorporating spatial priors:

Coordinate-based embedding models (e.g., CoordConv, UNETR with positional embeddings)

Body Part Regression integration

LocBAM: Our novel 1D attention-based module to inject axis-wise anatomical awareness into convolutional encoders.

Originality & Framework
All code was written from scratch using the MONAI framework and standard PyTorch. No proprietary or third-party segmentation code was reused beyond standard open-source libraries.

# Inspiration
LocBAM module was conceptually inspired by the HANet architecture for height-aware attention in 2D scene parsing:

HANet GitHub: https://github.com/lhc1224/HANet
License: Creative Commons BY-NC 4.0

# License
This repository is licensed under the Apache License 2.0, permitting free use, modification, and distribution with attribution. See the LICENSE file for full terms.