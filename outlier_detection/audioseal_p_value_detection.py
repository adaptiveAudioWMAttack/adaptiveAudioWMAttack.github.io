import os
import sys
import torch
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audioseal import AudioSeal
import librosa
import math
from scipy.stats import norm

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)

directory_path = './audioseal_creation/Librispeech_262_baseline'  # Update this path
file_extensions = ['.wav', '.flac', '.mp3']

audio_files = [file for file in os.listdir(directory_path) if any(file.endswith(ext) for ext in file_extensions)]

num1 = len(audio_files)
num2 = 0

data_outliers = []

def pvalue_two_tailed(x: float, mean: float, std: float) -> (float, float):
    # handle degenerate std
    if std <= 0 or not np.isfinite(std):
        # If std is invalid, treat deviations as infinitely unlikely (p=0) unless exactly equal
        if x == mean:
            return 1.0, 0.0
        else:
            return 0.0, math.copysign(float('inf'), x - mean)
    z = (x - mean) / std
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    # numerical guard
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return p, z

def is_outlier_pvalue(x: float, mean: float, std: float, alpha: float = 0.05):
    p, z = pvalue_two_tailed(x, mean, std)
    return (p < alpha), p, z
# ------------------------------------------------------------------

# Your fitted distribution params
mean_below_05 = 0.23234380781650543
std_below_05 = 0.031044764444231987
mean_above_05 = 0.773042619228363
std_above_05 = 0.029324112460017204

alpha = 0.003  # significance level for two-tailed test

defense_success_count = 0

for audio_file in audio_files:
    print("Processing:", audio_file)
    file_path = os.path.join(directory_path, audio_file)

    wav, sr = librosa.load(file_path, sr=None)

    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)
    
    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    result, message = detector(wav)
    message = message.squeeze().detach().cpu().numpy()
    
    print("result[:, 1 , :]:", result[:, 1 , :])
    print("message:", message)
    
    outlier_detected = False
    for i, bit in enumerate(message):
        if bit < 0.5:
            is_out, p, z = is_outlier_pvalue(bit, mean_below_05, std_below_05, alpha)
            side = "below_0.5"
            mu_used, std_used = mean_below_05, std_below_05
        else:
            is_out, p, z = is_outlier_pvalue(bit, mean_above_05, std_above_05, alpha)
            side = "above_0.5"
            mu_used, std_used = mean_above_05, std_above_05

        if is_out:
            data_outliers.append([
                audio_file, i, float(bit), float(z), float(p),
                side, float(mu_used), float(std_used), alpha
            ])
            outlier_detected = True
    
    if outlier_detected:
        defense_success_count += 1

    num2 += 1
    print("num1/num2:", num1, "/", num2)

# print(data_outliers)

# Save outliers to CSV
# if data_outliers:
#     df_outliers = pd.DataFrame(
#         data_outliers,
#         columns=[
#             "Audio File", "Bit Index", "Probability",
#             "z_score", "p_value", "bucket", "mean_used", "std_used", "alpha"
#         ]
#     )
    # os.makedirs("./results", exist_ok=True)
    # df_outliers.to_csv("./results/outlier_results.csv", index=False)
    # print("Outlier results saved to ./results/outlier_results.csv")

# Calculate and print defense success rate
defense_success_rate = defense_success_count / num1 if num1 > 0 else 0.0
print(f"Defense Success Rate: {defense_success_rate:.4f} ({defense_success_count}/{num1})")
