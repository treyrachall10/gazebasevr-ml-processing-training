# GazeBaseVR Machine Learning Project

> **Disclaimer**  
> I'm not a professional ML engineer — this project is part of my personal learning journey in AI and machine learning.
> 
> The code may not follow best practices, and I'm still experimenting and improving. Feedback is welcome.
> 
> The models and processing scripts have only been written to work with the round 1 data from the GazeBaseVR dataset.
> 
> The code to save the models has not been implemented here. So feel free to add that code as you wish.
> 
> This script will only work if you have a proper GPU for training.

---

## Overview

This project focuses on **cleaning, preprocessing**, and **training three different AI models** adapted from the paper **Evaluating The Long-Term Viability Of Eye-Tracking For Continuous Authentication In Virtual Reality** on the **GazeBaseVR dataset**, which contains eye-tracking data collected in virtual reality environments.

The main goal is to predict or classify participant identity using gaze behavior over time, contributing to research around **biometrics, user authentication, and human behavior in VR**.

The paper **Evaluating The Long-Term Viability Of Eye-Tracking For Continuous Authentication In Virtual Reality** provided the architecture for the models. However, they did not provide the code for the models. So I have created this public repo to save people some time when creating these models.

---

## What This Project Does

- Cleans and filters raw GazeBaseVR eye-tracking data
- Normalizes and windows the data into consistent segments  
- Trains 3 machine learning models:
  - XGBoost (for structured/tabular data)
  - Transformer Encoder (for temporal sequences)
  - DenseNet1D following the EKYT architecture (for spatial-temporal patterns)
- Supports GPU acceleration for faster training
- Includes TensorBoard integration for training visualization

---

## Why I Did This

I'm currently participating in a REU. My project focuses on creating a dataset with kids eye gaze data that closely aligns with the **GazeBaseVR** dataset.
The idea is that if I can recreate the models that other papers used to validate the dataset, then my dataset could be slid into the models to be validated after it is done being created.

This project helped me:
- Practice real-world data preprocessing
- Understand model performance on gaze-based inputs
- Learn about different ML architectures and their strengths/weaknesses

---

## Project Flow

```text
Start
 │
 ▼
User runs one of the model training scripts:
  python transformer_encoder.py --src <raw_data> --round_1_dir <round1_dir> --norm_dir <normalized_data>
  OR
  python xgboost_model.py --src <raw_data> --round_1_dir <round1_dir> --norm_dir <normalized_data>
  OR
  python densenet_model.py --src <raw_data> --round_1_dir <round1_dir> --norm_dir <normalized_data>
 │
 ▼
Each script automatically:
  1. Normalizes file paths and creates output directories if needed
  2. Calls getXY() from preprocess_files.py
     └── Moves round 1 files
     └── Cleans invalid or short recordings
     └── Normalizes gaze/base positions
     └── Splits data into windows
     └── Returns X (features) and Y (labels)
 │
 ▼
The selected model is trained on the processed data
  └── Prints validation accuracy
  └── Logs training loss to TensorBoard (for deep learning models)
 │
 ▼
End
```
