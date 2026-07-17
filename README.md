# 🫁Deep Learning-Based 3D Segmentation of Pulmonary Nodules for Lung Cancer Detection
An end-to-end deep learning framework for automated pulmonary nodule segmentation from volumetric chest CT scans using a customized 3D U-Net pipeline.

Developed in PyTorch and MONAI as part of a Master's research project in Data Science, this repository includes preprocessing, training, scan-level inference, false-positive reduction, and comprehensive evaluation on the LIDC-IDRI dataset.

![Pipeline](images/fullpipelinediagram.png)

## Overview

Pulmonary nodule segmentation is a fundamental task in computer-aided diagnosis systems for lung cancer screening. Accurate delineation of nodules enables volumetric analysis, treatment monitoring, and downstream clinical decision support. However, reliable segmentation remains challenging due to variations in nodule size, shape, texture, and CT acquisition parameters.

This project presents an end-to-end deep learning framework for automatic pulmonary nodule segmentation from volumetric CT scans. The framework combines standardized medical image preprocessing, 3D convolutional neural networks, sliding-window scan inference, false-positive reduction, and quantitative evaluation into a reproducible pipeline.

Rather than focusing solely on model training, this repository emphasizes the complete engineering workflow required to develop and evaluate a practical medical image segmentation system.
## Motivation

Deep learning has significantly advanced medical image analysis, yet pulmonary nodule segmentation remains challenging because of highly variable nodule morphology, limited annotation agreement among radiologists, and the scarcity of subtle lesion types such as ground-glass opacities.

The objective of this project was not only to train a segmentation model, but to design and evaluate a complete workflow capable of handling real-world CT data. Special attention was given to robust preprocessing, scan-level inference, annotation uncertainty, and quantitative evaluation under clinically relevant conditions.

The resulting framework was developed as a Master's thesis in Data Science and demonstrates the practical implementation of modern deep learning techniques for medical image segmentation.