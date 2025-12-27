![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# DRiVE: Dynamic Recognition in VEhicles using snnTorch
https://arxiv.org/pdf/2502.10421

## 📌 Project Summary
This repository implements a Spiking Neural Network (SNN) used to classify images between vehicles and nonvehicles as researched in [this article](https://arxiv.org/pdf/2502.10421). The work provides a baseline for image classification for an emerging type of computing--neuromorphic computing--which aims to conserve energy and mimic the human brain utilizing models based in spiking. It utilizes a new extensive library called snnTorch allowing SNNs to be made easily in Python.

## 🎯 Motivation
Traditional image detection systems, such as those based in CNNs, are highly computationally expensive. Spiking models from neuromorphic computing such as SNNs have been increasingly studied in applications for real-time capabilities while significantly reducing energy needs, providing vital function in systems such as IoT (Internet of Things) devices. This study in particular focuses on utilizing SNNs for vehicle image detection which is critical in autonomous vehicle systems.

## 🧩 Novelty
Unlike traditional SNNs that process time-based data streams, DRiVE encodes static grayscale images as spike trains and processes them through temporal dynamics.  
This approach:
- Converts spatial features into temporal patterns
- Leverages membrane potential dynamics for feature extraction
- Achieves CNN-comparable accuracy with significantly lower energy

## 🧠 Methodology
1. **Dataset**: Uses the [Vehicle Detection Image Dataset](https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set) using 1000 out of 8000 images per class (vehicle vs. nonvehicle). These are pre-processed using greyscale, normalization and resizing.
2. **SNN**:  
- Input layer: Normalizes input data from images
encoded as spike trains.
- Two Hidden Layers: Utilize Leaky Integrate-andFire (LIF) neurons with surrogate gradients.
- Output Layer: Produces spike outputs
corresponding to predicted classes (vehicle or nonvehicle).
3. **Evaluation**:  
   - Accuracy, ROC curve, and loss curve

**Future Work**:
1. **Real-time video** - Leverage SNN temporal strengths for vehicle videos rather than static images
2. **Model compression** - Further reduce parameters and energy
3. **Hybrid architectures** - Combine CNN feature extraction with SNN inference

## 🌍 Impact
1. **Edge deployment opportunity** - SNNs enable battery-powered vision systems
2. **Framework maturity** - snnTorch makes SNN development accessible
