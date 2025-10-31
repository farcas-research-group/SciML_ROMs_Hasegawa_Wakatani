# Learning physics-based reduced models from data for the Hasegawa-Wakatani equations

https://arxiv.org/abs/2401.05972

## Abstract

This paper focuses on the construction of non-intrusive Scientific Machine Learning (SciML) Reduced-Order Models (ROMs) for nonlinear, chaotic plasma turbulence simulations. In particular, we propose using Operator Inference (OpInf) to build low-cost physics-based ROMs from data for such simulations. As a representative example, we focus on the Hasegawa-Wakatani (HW) equations used for modeling two-dimensional electrostatic drift-wave plasma turbulence. For a comprehensive perspective of the potential of OpInf to construct accurate ROMs for this model, we consider a setup for the HW equations that leads to the formation of complex, nonlinear and self-driven dynamics, and perform two sets of experiments. We first use the data obtained via a direct numerical simulation of the HW equations starting from a specific initial condition and train OpInf ROMs for predictions beyond the training time horizon. In the second, more challenging set of experiments, we train ROMs using the same data set as before but this time perform predictions for six other initial conditions. Our results show that the OpInf ROMs capture the important features of the turbulent dynamics and generalize to new and unseen initial conditions while reducing the evaluation time of the high-fidelity model by up to six orders of magnitude in single-core performance. In the broader context of fusion research, this shows that non-intrusive SciML ROMs have the potential to drastically accelerate numerical studies, which can ultimately enable tasks such as the design and real-time control of optimized fusion devices.

## How to download the data

The data is available for download. Using `rclone`, the following configuration allows you to download the data:
```
rclone config create opinf-for-hw webdav url https://datashare.mpcdf.mpg.de/public.php/webdav/ user Us0faUb9pbhtfdw
rclone config create opinf-for-hw-overlay chunker remote opinf-for-hw: chunk_size 2G hash_type none
```
To download the data into the `data/` directory, simply run:
```
rclone copy opinf-for-hw-overlay: data/ --progress
```

## How to install

Installing the code opinf library used should be straight forward, simply run `pip install .` or equivalent commands for `uv` or `anaconda`.

## How to run

The code is split into several parts. In the `opinf_for_hw/` directory, you can find the library code that is used in several places and, most importantly, the global configuration file. With this file you can control whether you want to do predictions beyond the training data or predictions for multiple initial conditions. In addition, you can set the reduced rank there.

The different steps of the OpInf procedure can be found in `scripts_c1_*/`. Each folder contains the code for the basis computation, the learning of the ROM, and the predictions.

Finally, the visualization folder contains scripts creating all figures present in the paper.

