# Learning to Evade: Statistical Learning-Based Adaptive Attacks Towards Audio Watermarking

This paper has been submitted to Usenix Security 2026 Cycle 1 for peer reviewing.

# How to Use
## Dependencies
```
conda create -n AWM python=3.9.21
conda activate AWM
pip install -r requirements.txt
pip install librosa==0.9.1
```

## Install ViSQOL
Please visit this website https://github.com/google/visqol

# Dataset

For now, we only provide the code and data for the watermark replacement attack. The complete code will be published after the paper is accepted, or can be released upon request.  

The dataset is available at [[link]](https://drive.google.com/drive/folders/1od-PvwZOv4Kz2HTnkbkYIon1qT4Oxiqy?usp=sharing). You can reproduce the experiments using the data provided there.  

- **audioseal_replacement**: Data for the watermark replacement attack using the **AudioSeal** method.  
  - *baseline*: generated from AudioMarkBench  
  - *my_method*: generated from AWM  
  - *my_method_optimization*: generated from AWM (+opt)  

- **timbre_replacement**: Data for the watermark replacement attack using the **Timbre** method.  
  - *baseline*: generated from AudioMarkBench  
  - *my_method*: generated from AWM  
  - *my_method_optimization*: generated from AWM (+opt)  

- **dataset_audioseal**: Benign data (clean and watermarked) using the **AudioSeal** method.  

- **data_timbre**: Benign data (clean and watermarked) using the **Timbre** method.  

- **checkpoint**: Model checkpoint for **Timbre**. 

# Defender -- Outlier Detection
