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

directory_path = './dataset_timbre/timbre_librispeech_testing_262'

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

# ---- Normal parameters (your fitted stats) ----
mean = -0.00012657273327931762
std  = 0.13281798362731934

def pvalue_two_tailed(x: float, mu: float, sd: float):
    # returns (p_value, z_score)
    if sd <= 0 or not np.isfinite(sd):
        return ((1.0, 0.0) if x == mu else (0.0, math.copysign(float('inf'), x - mu)))
    z = (x - mu) / sd
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    # clamp for safety
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return p, z

alpha = 0.011  # significance level for two-tailed test

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
        p, z = pvalue_two_tailed(value, mean, std)
        if p < alpha:
            outlier_results.append([audio_file, i, float(value), float(z), float(p), float(mean), float(std), alpha])
            outlier_detected = True

    if outlier_detected:
        defense_success_count += 1

    num2 += 1
    print("num1/num2:", num1, "/", num2)

# Save outliers to CSV (now includes z and p)
# os.makedirs('./results', exist_ok=True)
# outlier_df = pd.DataFrame(
#     outlier_results,
#     columns=['Audio File', 'Bit Index', 'Value', 'z_score', 'p_value', 'mean_used', 'std_used', 'alpha']
# )
# outlier_df.to_csv('./results/outlier_results.csv', index=False)

# Defense success rate
defense_success_rate = defense_success_count / num1 if num1 > 0 else 0
print(f"Defense Success Rate: {defense_success_rate * 100:.4f}% ({defense_success_count}/{num1})")