import os
import argparse
import numpy as np
import torch
import torchaudio
from audioseal import AudioSeal
import yaml
from timbre.model.conv2_mel_modules import Decoder
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import time
import torchaudio.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
import copy
import torch.nn.functional as F
import math
import random


def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with audioseal")

    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")

    parser.add_argument("--iter", type=int, default=1000, help="Number of iterations for the attack")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the attack")

    parser.add_argument("--whitebox_folder", type=str, default="whitebox_debug",
                        help="Folder to save the whitebox attack results")

    parser.add_argument("--tau", type=float, default=0.95, help="Threshold for the detector")

    parser.add_argument("--attack_bitstring", action="store_true", default=True,
                        help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--model", type=str, default='audioseal', choices=['audioseal', 'timbre'],
                        help="Model to be attacked")
    parser.add_argument("--dataset", type=str, default="librispeech", help="Dataset to use for the attack")

    print("Arguments: ", parser.parse_args())
    return parser.parse_args()


def api_visqol():
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    return api


class WatermarkDetectorWrapper():
    def __init__(self, model, message, on_bitstring, model_type, threshold, device):
        self.model = model
        self._device = device
        self.message = message.to(self._device)
        self.on_bitstring = on_bitstring
        self.model.to(self._device)
        self.model_type = model_type
        self.threshold = threshold
        if model_type == 'timbre':
            self.bwacc = self.bwacc_timbre
            self.optimize_loss = self.optimize_loss_timbre
        elif model_type == 'audioseal':
            self.bwacc = self.bwacc_audioseal
            self.optimize_loss = self.optimize_loss_audioseal
    
    def computer_spec_loss(self, target_waveform, predicted_waveform, device):
        sample_rate = 16000
        transform = transforms.Spectrogram(n_fft=800).to(device)
        target_specgram = transform(target_waveform).to(device)
        predicted_specgram = transform(predicted_waveform).to(device)
        
        target_specgram_flatten = target_specgram.flatten()
        predicted_specgram_flatten = predicted_specgram.flatten()
        
        softmax = nn.Softmax()
        output_target_specgram_flatten = softmax(target_specgram_flatten) * target_specgram_flatten
        output_predicted_specgram_flatten = softmax(predicted_specgram_flatten) * predicted_specgram_flatten

        l1loss = nn.L1Loss(reduction="sum")
        spec_loss = l1loss(output_target_specgram_flatten, output_predicted_specgram_flatten)
        return spec_loss

    def bwacc_audioseal(self, signal):
        result, msg_decoded = self.model.detect_watermark(signal)
        if self.on_bitstring:
            if msg_decoded is None:
                return torch.zeros(1)
            else:
                bitacc = 1 - torch.sum(torch.abs(self.message - msg_decoded)) / self.message.numel()
                return bitacc
        else:
            return result

    def bwacc_timbre(self, signal):  # signal is tensor on gpu
        payload = self.model.test_forward(signal)
        target_message = self.message.detach()
        target_message = target_message * 2 - 1
        payload = payload.to(self._device)
        bitacc = (payload >= 0).eq(target_message >= 0).sum().float() / target_message.numel()
        return bitacc
    
    def optimize_loss_audioseal(self, original_watermark_audio, new_watermark_audio, gt_message_input):
        MSEloss = nn.MSELoss(reduction='sum')
        l1loss = nn.L1Loss(reduction='sum')
        BCEloss = nn.BCELoss()
        
        _ , new_watermark_audio_message = self.model(new_watermark_audio)
        new_watermark_audio_message = new_watermark_audio_message.squeeze()

        results, messages = self.model(new_watermark_audio)
        class_1 = results[:, 1, :]
        gt_class = torch.full_like(class_1, 0.9999)
        m = nn.Sigmoid()
        noise_class = m(class_1)

        a = original_watermark_audio.size(2) / 160
        b = original_watermark_audio.size(2) / 1600
        
        wav_loss = l1loss(original_watermark_audio, new_watermark_audio)
        computer_spec_loss = self.computer_spec_loss(original_watermark_audio, new_watermark_audio, device=self._device)
        message_loss = MSEloss(gt_message_input, new_watermark_audio_message)
        position_loss = BCEloss(gt_class, noise_class)
        loss = wav_loss + computer_spec_loss + a * message_loss + b * position_loss
        print("gt_message_input: ", gt_message_input)
        print("new_watermark_audio_message: ", new_watermark_audio_message)
        print("wav_loss: ", wav_loss)
        print("computer_spec_loss: ", computer_spec_loss)
        print("message_loss", message_loss)
        print("position_loss: ", position_loss)
        print("loss: ", loss)
        return loss
    
    def optimize_loss_timbre(self, original_watermark_audio, new_watermark_audio, gt_message_input):
        MSEloss = nn.MSELoss(reduction='sum')
        l1loss = nn.L1Loss(reduction='sum')
        
        new_watermark_audio_message = self.model.test_forward(new_watermark_audio).to(self._device).squeeze()

        a = original_watermark_audio.size(2) / 160
        
        wav_loss = l1loss(original_watermark_audio, new_watermark_audio)
        computer_spec_loss = self.computer_spec_loss(original_watermark_audio, new_watermark_audio, device=self._device)
        message_loss = MSEloss(gt_message_input, new_watermark_audio_message)
        loss = wav_loss + computer_spec_loss + a * message_loss
        print("gt_message_input: ", gt_message_input)
        print("new_watermark_audio_message: ", new_watermark_audio_message)
        print("wav_loss: ", wav_loss)
        print("computer_spec_loss: ", computer_spec_loss)
        print("message_loss", message_loss)
        print("loss: ", loss)
        return loss

def optimize_audio_quality(detector, original_watermark_audio, adv_signal, tensor_pert, gt_message_input, snr, args):
    best_snr = copy.deepcopy(snr)
    best_snr_copy = copy.deepcopy(snr)
    best_adv_signal = torch.zeros_like(original_watermark_audio)
    if isinstance(gt_message_input, torch.Tensor):
        gt_message_input = gt_message_input.detach().squeeze()
    for param in detector.model.parameters():
        param.requires_grad = False
    if args.model == "audioseal":
        bwacc = detector.bwacc(original_watermark_audio)
        index_flag = list(range(16))
        percentage = int(16.0 - bwacc * 16.0)
        if percentage < 4:
            percentage = 4
        if percentage > 8:
            percentage = 8
    original_watermark_audio.requires_grad = False
    tensor_pert.requires_grad = True
    optimizer = optim.Adam([tensor_pert], lr=args.lr)
    num = 0
    best_adv_num = 0
    for _ in range(args.iter):
        detector.model.train()
        optimizer.zero_grad()
        new_watermark_audio = original_watermark_audio + tensor_pert
        loss = detector.optimize_loss(original_watermark_audio, new_watermark_audio, gt_message_input)
        loss.backward()
        optimizer.step()
        detector.model.eval()
        with torch.no_grad():
            num = num + 1
            new_watermark_audio = original_watermark_audio + tensor_pert
            snr = 10 * torch.log10(torch.mean(original_watermark_audio ** 2) / torch.mean(tensor_pert ** 2))
            print(f'snr: {snr}, acc: {detector.bwacc(new_watermark_audio)}, num: {num}')
            if snr > best_snr and detector.bwacc(new_watermark_audio) > args.tau:
                if args.model == 'timbre':
                    noise_message = detector.model.test_forward(new_watermark_audio).squeeze()
                    threshold_set = 0
                    one_right = 1.11756
                    one_left = 0.89621
                    zero_right = -0.86451
                    zero_left = -1.13026
                    noise_message = detector.model.test_forward(new_watermark_audio).squeeze()
                    cnt = len(noise_message)
                elif args.model == 'audioseal':
                    threshold_set = 0.5
                    one_right = 0.82533
                    one_left = 0.7265
                    zero_right = 0.28762
                    zero_left = 0.17669
                    _ , noise_message = detector.model(new_watermark_audio)
                    noise_message = noise_message.squeeze()
                    cnt = percentage
                for index in range(0, len(gt_message_input)):
                    if gt_message_input[index] >= threshold_set:
                        if gt_message_input[index] < one_left or gt_message_input[index] > one_right:
                            cnt -= 1
                    if gt_message_input[index] < threshold_set:
                        if gt_message_input[index] < zero_left or gt_message_input[index] > zero_right:
                            cnt -= 1
                
                for index in range(0, len(noise_message)):
                    if gt_message_input[index] >= threshold_set:
                        if noise_message[index] > one_left and noise_message[index] < one_right:
                            cnt -= 1
                    if gt_message_input[index] < threshold_set:
                        if noise_message[index] > zero_left and noise_message[index] < zero_right:
                             cnt -= 1
                    print("cnt: ", cnt)
                    if cnt <= 0:
                        best_snr = snr
                        best_adv_signal = new_watermark_audio.clone().detach()
                        best_adv_num = num
    # exit()
    if best_snr_copy == best_snr:
        return adv_signal.clone().detach(), None
    return best_adv_signal, best_adv_num

def decode_audio_files_perturb_whitebox(model, output_dir, args, device):
    # 原始的音频文件
    watermarked_files = os.listdir(os.path.join(output_dir, 'timbre_librispeech_testing_262'))
    progress_bar = tqdm(enumerate(watermarked_files), desc="Decoding Watermarks under whitebox attack")
    save_path = os.path.join(output_dir, args.whitebox_folder)
    os.makedirs(save_path, exist_ok=True)
    # 攻击后的音频文件
    adv_signal_files = os.listdir(os.path.join(output_dir, 'Librispeech_262_my_method'))
    visqol = api_visqol()
    for file_num, watermarked_file in progress_bar:
        idx = os.path.splitext(watermarked_file)[0]
        base_name = watermarked_file.split("_", 1)[-1]

        base_no_ext = os.path.splitext(base_name)[0]
        if args.dataset == 'librispeech':
            watermarked_file_new = next((f for f in adv_signal_files if f.startswith(base_no_ext + "_")), None)
            if watermarked_file_new is None:
                print(f"Skipping {idx}: No matching file found.")
                continue
            waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'timbre_librispeech_testing_262', watermarked_file))
            adv_signal, sample_rate = torchaudio.load(os.path.join(output_dir, 'Librispeech_262_my_method', watermarked_file_new))
        elif args.dataset == 'audiomark':
            if args.model == "timbre":
                idx = watermarked_file.split("_", 1)[0] + "_" + base_no_ext
                base_no_ext_audiomark = watermarked_file.split("_", 1)[0] + "_" + base_no_ext
                watermarked_file_new = next((f for f in adv_signal_files if f.startswith(base_no_ext_audiomark + "_")), None)
                if watermarked_file_new is None:
                    print(f"Skipping {idx}: No matching file found.")
                    continue
                waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'timbre_audiomark_16khz', watermarked_file))
                adv_signal, sample_rate = torchaudio.load(os.path.join(output_dir, 'audiomark_16khz_my_method', watermarked_file_new))
            else:
                idx = os.path.splitext(base_name)[0]
                watermarked_file_new = next((f for f in adv_signal_files if f.startswith(base_no_ext + "_")), None)
                waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'timbre_audiomark_16khz', watermarked_file))
                adv_signal, sample_rate = torchaudio.load(os.path.join(output_dir, 'audiomark_16khz_my_method', watermarked_file_new))
        elif args.dataset == 'gigaspeech':
            idx = watermarked_file.split("_", 1)[0] + "_" + base_no_ext
            base_no_ext_gig = watermarked_file.split("_", 1)[0] + "_" + base_no_ext
            watermarked_file_new = next((f for f in adv_signal_files if f.startswith(base_no_ext_gig + "_")), None)
            if watermarked_file_new is None:
                print(f"Skipping {idx}: No matching file found.")
                continue
            waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'timbre_gigaspeech_xs', watermarked_file))
            adv_signal, sample_rate = torchaudio.load(os.path.join(output_dir, 'gigaspeech_xs_my_method', watermarked_file_new))

        # waveform原始的音频，adv_signal攻击后的音频
        waveform = waveform.to(device=device)
        waveform = waveform.unsqueeze(0)

        adv_signal = adv_signal.to(device=device)
        adv_signal = adv_signal.unsqueeze(0)

        if args.model == "audioseal":
            _ , attacked_message = model.detect_watermark(adv_signal)
            _ , gt_message_input = model(adv_signal)
            gt_message_input = gt_message_input.squeeze()
        elif args.model == "timbre":
            gt_message_input = model.test_forward(adv_signal).squeeze()
            attacked_message = model.test_forward(adv_signal).squeeze()
            attacked_message = (attacked_message > 0).float()
        
        attack_bitstring = True

        detector = WatermarkDetectorWrapper(model, attacked_message, attack_bitstring, args.model, args.tau, device)
        if args.model == "audioseal":
            tensor_pert = adv_signal - waveform
            snr = 10 * torch.log10(torch.mean(waveform ** 2) / torch.mean(tensor_pert ** 2))
            adv_signal_optimize, num_optimize = optimize_audio_quality(detector, waveform, adv_signal, tensor_pert, gt_message_input, snr, args)
        elif args.model == "timbre":
            tensor_pert = adv_signal - waveform
            snr = 10 * torch.log10(torch.mean(waveform ** 2) / torch.mean(tensor_pert ** 2))
            adv_signal_optimize, num_optimize = optimize_audio_quality(detector, waveform, adv_signal, tensor_pert, gt_message_input, snr, args)
        
        if num_optimize == None:
            num_optimize = -1
        if adv_signal_optimize != None:
            adv_signal = adv_signal_optimize
        '''save to log file'''
        filename = os.path.join(save_path, f'whitebox.csv')
        log = open(filename, 'a' if os.path.exists(filename) else 'w')
        log.write('idx, query, acc, snr, visqol\n')
        acc = detector.bwacc(adv_signal)
        snr = 10 * torch.log10(torch.sum(waveform ** 2) / torch.sum((adv_signal - waveform) ** 2))
        visqol_score = visqol.Measure(np.array(waveform.squeeze().detach().cpu(), dtype=np.float64),
                                      np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)).moslqo
        print(f'idx: {idx}, query_optimize: {num_optimize} acc: {acc:.3f}, snr: {snr:.3f}, visqol: {visqol_score}')
        log.write(f'{idx}, {num_optimize}, {acc}, {snr}, {visqol_score}\n')
        torchaudio.save(os.path.join(save_path,
                                         f"{idx}_tau{args.tau}_query_optimize{num_optimize}_snr{snr:.3f}_acc{acc:.3f}_visqol{visqol_score}.wav"),
                            adv_signal.squeeze(0).detach().cpu(), sample_rate)

def main():
    args = parse_arguments()

    np.random.seed(42)
    torch.manual_seed(42)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.model == 'audioseal':
        model = AudioSeal.load_detector("audioseal_detector_16bits").to(device=device)
        output_dir = 'optimization_audioseal'
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'timbre':
        process_config = yaml.load(open("white-box/timbre/config/process.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open("white-box/timbre/config/model.yaml", "r"), Loader=yaml.FullLoader)
        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
        msg_length = 16
        detector = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim,
                           nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
        checkpoint = torch.load(
            './white-box/timbre/checkpoint/none-conv2_ep_15_2025-03-09_18_35_03.pth.tar')
        detector.load_state_dict(checkpoint['decoder'], strict=False)
        detector.eval()
        model = detector
        output_dir = 'optimization_timbre'
        os.makedirs(output_dir, exist_ok=True)

    decode_audio_files_perturb_whitebox(model, output_dir, args, device)


if __name__ == "__main__":
    main()
