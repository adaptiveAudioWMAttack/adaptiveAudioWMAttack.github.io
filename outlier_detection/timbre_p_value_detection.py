import os
import torch
import soundfile as sf
import numpy as np
import pandas as pd
import yaml
from timbre.model.conv2_mel_modules import Decoder
import librosa
import math
from scipy.stats import norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

directory_path = './timbre_creation/Librispeech_262_baseline'

process_config = yaml.load(open("outlier_detection/timbre/config/process.yaml", "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open("outlier_detection/timbre/config/model.yaml", "r"), Loader=yaml.FullLoader)
win_dim = process_config["audio"]["win_len"]
embedding_dim = model_config["dim"]["embedding"]
nlayers_decoder = model_config["layer"]["nlayers_decoder"]
attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
msg_length = 16
detector = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim,
                   nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
checkpoint = torch.load('./outlier_detection/timbre/checkpoint/none-conv2_ep_15_2025-03-09_18_35_03.pth.tar')
detector.load_state_dict(checkpoint['decoder'], strict=False)
detector.eval()
model = detector

file_extensions = ['.wav', '.flac', '.mp3']
all_files = os.listdir(directory_path)
audio_files = [file for file in all_files if any(file.endswith(ext) for ext in file_extensions)]

defense_success_count = 0
outlier_results = []

def pvalue_two_tailed(x: float, mean: float, std: float):
    # returns (p_value, z_score)
    z = (x - mean) / std
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    # clamp for numerical safety
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return p, z

# ---- Normal model parameters per bucket ----
mean_neg = -1.0121878385543823
std_neg  = 0.06052171811461449
mean_pos =  1.0138801336288452
std_pos  =  0.06031925976276398

# Significance level (two-tailed)
alpha = 0.003

num1 = len(audio_files)
num2 = 0

for audio_file in audio_files:
    print("Processing:", audio_file)
    file_path = os.path.join(directory_path, audio_file)
    waveform, sample_rate = librosa.load(file_path, sr=None)
    waveform = torch.from_numpy(waveform).float().to(device=device)
    waveform = waveform.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        message = model.test_forward(waveform).to(device).squeeze()
    message = message.detach().cpu().numpy()
    print("message: ", message)

    outlier_detected = False
    for i, value in enumerate(message):
        if value < 0:
            mu, sd, bucket = mean_neg, std_neg, "neg"
        else:
            mu, sd, bucket = mean_pos, std_pos, "pos"

        p, z = pvalue_two_tailed(value, mu, sd)
        if p < alpha:
            outlier_results.append([
                audio_file, i, float(value), float(z), float(p), bucket, float(mu), float(sd), alpha
            ])
            outlier_detected = True

    if outlier_detected:
        defense_success_count += 1

    num2 += 1
    print("num1/num2:", num1, "/", num2)

# Save outliers to CSV (now includes z and p)
# os.makedirs('./results', exist_ok=True)
# outlier_df = pd.DataFrame(
#     outlier_results,
#     columns=['Audio File', 'Bit Index', 'Value', 'z_score', 'p_value', 'bucket', 'mean_used', 'std_used', 'alpha']
# )
# outlier_df.to_csv('./results/timbre_new_outlier_results.csv', index=False)
# print("Outlier results saved to ./results/timbre_new_outlier_results.csv")

# Defense success rate
defense_success_rate = defense_success_count / num1 if num1 > 0 else 0
print(f"Defense Success Rate: {defense_success_rate * 100:.4f}% ({defense_success_count}/{num1})")
