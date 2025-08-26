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

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)

directory_path = './audioseal_remove/Librispeech_262_my_method_optimization'
file_extensions = ['.wav', '.flac', '.mp3']

audio_files = [file for file in os.listdir(directory_path) if any(file.endswith(ext) for ext in file_extensions)]

num1 = len(audio_files)
num2 = 0

data_outliers = []

def pvalue_two_tailed(x: float, mean: float, std: float):
    # returns (p_value, z_score)
    if std <= 0 or not np.isfinite(std):
        # Degenerate case: if std invalid, treat exact mean as p=1 else p=0 (with inf z)
        return (1.0, 0.0) if x == mean else (0.0, math.copysign(float('inf'), x - mean))
    z = (x - mean) / std
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    # numerical safety
    p = max(0.0, min(1.0, p))
    return p, z

def is_outlier_pvalue(x: float, mean: float, std: float, alpha: float = 0.05):
    p, z = pvalue_two_tailed(x, mean, std)
    return (p < alpha), p, z
# -------------------------------------------------------------------

# Your fitted normal parameters
mean = 0.5012016296386719
std  = 0.018524717539548874
alpha = 0.003  # significance level

defense_success_count = 0

for audio_file in audio_files:
    print("Processing:", audio_file)
    file_path = os.path.join(directory_path, audio_file)
    # wav, sr = sf.read(file_path)
    wav, sr = librosa.load(file_path, sr=None)

    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float().to(device)

    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    with torch.no_grad():
        result, message = detector(wav)
    message = message.squeeze().detach().cpu().numpy()

    print("result[:, 1 , :]:", result[:, 1 , :])
    print("message:", message)

    outlier_detected = False
    for i, bit in enumerate(message):
        is_out, p, z = is_outlier_pvalue(bit, mean, std, alpha)
        if is_out:
            data_outliers.append([audio_file, i, float(bit), float(z), float(p), float(mean), float(std), alpha])
            outlier_detected = True

    if outlier_detected:
        defense_success_count += 1

    num2 += 1
    print("num1/num2:", num1, "/", num2)

# Save outliers to CSV
# if data_outliers:
#     os.makedirs("./results", exist_ok=True)
#     df_outliers = pd.DataFrame(
#         data_outliers,
#         columns=["Audio File", "Bit Index", "Probability", "z_score", "p_value", "mean_used", "std_used", "alpha"]
#     )
#     df_outliers.to_csv("./results/audioseal_new_outlier_results.csv", index=False)
#     print("Outlier results saved to ./results/audioseal_new_outlier_results.csv")

# Calculate and print defense success rate
defense_success_rate = defense_success_count / num1 if num1 > 0 else 0.0
print(f"Defense Success Rate: {defense_success_rate:.4f} ({defense_success_count}/{num1})")