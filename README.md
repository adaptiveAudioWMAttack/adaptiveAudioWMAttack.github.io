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

The example dataset is available at [[dataset]](https://drive.google.com/drive/folders/1od-PvwZOv4Kz2HTnkbkYIon1qT4Oxiqy?usp=sharing). You can reproduce the experiments using the data provided there. Here is the example for the watermark replacement.

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
```
python outlier_detection/audioseal_p_value_detection.py
python outlier_detection/timbre_p_value_detection.py
```

# Attack -- Example for Watermark Replacement
## AWM
Before running the command, (1) please put the data to the folder "attack_audioseal" or the folder "attack_timbre", and then (2) change dataset to your own data in line 308 and line 316.
```
python white-box/watermark_replacement.py --model audioseal --whitebox_folder attack_audioseal
python white-box/watermark_replacement.py --model timbre --whitebox_folder attack_timbre
```
## AWM (+opt)
Before running the command, please put the data to the folder "optimization_audioseal" or the folder "optimization_timbre", and then (2) change dataset to your own benign watermarked data in line 219 and line 234, and your AWM adversarial data in line 224 and line 235.
```
python white-box/optimization_watermark_replacement.py --model audioseal ----whitebox_folder optimization_audioseal --dataset librispeech
python white-box/optimization_watermark_replacement.py --model timbre ----whitebox_folder optimization_timbre --dataset librispeech
```
Our demo website [[demo]](https://adaptiveaudiowmattack.github.io/)
