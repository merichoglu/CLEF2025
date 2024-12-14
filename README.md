# CLEF2025: Political Orientation and Power Classification in Turkish Parliamentary Debates

This repository contains the code and resources for the project on classifying political orientation and power dynamics in the Grand National Assembly of Turkey's parliamentary debates. The work was conducted as part of the CENG463: Introduction to Natural Language Processing course.

## Project Overview

The project focuses on two primary classification tasks:

1. **Political Ideology Classification**: Determining whether a speaker's political stance is 'Left' or 'Right'.
2. **Power Identification**: Identifying whether a speaker belongs to the 'Coalition' (governing party) or the 'Opposition'.

To address these tasks, the project employs:

- **XLM-RoBERTa**: A multilingual transformer model fine-tuned for both classification tasks.
- **Llama-3.2:1b**: An open-source large language model utilized for inference.

The models were evaluated on both Turkish and English translations of the parliamentary speeches.

## Repository Contents

- `Classification.ipynb`: Jupyter Notebook detailing the data preprocessing, model training, and evaluation processes.
- `inference.py`: Script for performing inference using the trained models.
- `logs.txt`: Log file capturing the training and evaluation metrics.
- `README.md`: This document.

# Evaluation Results
Evaluation results are available here: https://wandb.ai/merichaliloglu-metu-middle-east-technical-university/ParliamentaryDebates?nw=nwusermerichaliloglu
