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

    parser.add_argument("--iter", type=int, default=10000, help="Number of iterations for the attack")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the attack")

    parser.add_argument("--whitebox_folder", type=str, default="whitebox_debug",
                        help="Folder to save the whitebox attack results")

    parser.add_argument("--tau", type=float, default=0.95, help="Threshold for the detector")

    parser.add_argument("--attack_bitstring", action="store_true", default=True,
                        help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--model", type=str, default='audioseal', choices=['audioseal', 'timbre'],
                        help="Model to be attacked")

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
    def __init__(self, model, message, target_message_true, on_bitstring, model_type, threshold, device):
        self.model = model
        self._device = device
        self.message = message.to(self._device)
        self.target_message_true = target_message_true.to(self._device)
        self.on_bitstring = on_bitstring
        self.model.to(self._device)
        self.model_type = model_type
        self.threshold = threshold
        if model_type == 'timbre':
            self.bwacc = self.bwacc_timbre
            self.get_loss = self.loss_timbre
        elif model_type == 'audioseal':
            self.bwacc = self.bwacc_audioseal
            self.get_loss = self.loss_audioseal

    def computer_melspec_loss(self, target_waveform, predicted_waveform, device):
        sample_rate = 16000
        transfrom = transforms.MelSpectrogram(sample_rate).to(device)
        target_mel_specgram = transfrom(target_waveform).to(device)
        predicted_mel_specgram = transfrom(predicted_waveform).to(device)
        MSEloss = nn.MSELoss(reduction='mean')
        melspec_loss = MSEloss(predicted_mel_specgram, target_mel_specgram)
        return melspec_loss
    
    def process_target_message(self, flag_index, target_message, gt_messages):
        message_length = len(target_message)
        if flag_index:
            for index in range(message_length):
                if index not in flag_index:
                    target_message[index] = gt_messages[index]
        else:
            for index in range(message_length):
                target_message[index] = gt_messages[index]
        return target_message

    def loss_audioseal(self, watermarked_signal, watermarked_signal_with_noise, gt_messages_input, flag_index, index_flag_length):
        mel_spec_loss = self.computer_melspec_loss(watermarked_signal, watermarked_signal_with_noise, device=self._device)
        MSEloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        BCEloss = nn.BCELoss()
        wav_loss = l1loss(watermarked_signal_with_noise, watermarked_signal)
        
        _ , watermarked_signal_message = self.model(watermarked_signal)
        watermarked_signal_message = watermarked_signal_message.squeeze()
        noise_results, noise_messages = self.model(watermarked_signal_with_noise)
        gt_results, _ = self.model(watermarked_signal)
        target_message = copy.deepcopy(self.target_message_true)
        gt_messages = gt_messages_input.clone().detach()
        noise_messages = noise_messages.float()
        target_message = target_message.float()
        
        # 判断watermark不一样的地方
        target_message = self.process_target_message(flag_index, target_message, gt_messages)
        
        message_loss = MSEloss(noise_messages.squeeze(), target_message)
        noise_class = noise_results[:, 1, :]
        gt_class = gt_results[:, 1, :]
        position_loss = BCEloss(gt_class, noise_class)
        loss = wav_loss + index_flag_length * message_loss + position_loss + mel_spec_loss
        print("==============================================")
        print("noise_messages.squeeze(): ", noise_messages.squeeze())
        print("target_message", target_message)
        print("wav_loss: ", wav_loss)
        print("message_loss: ", message_loss)
        print("position_loss: ", position_loss)
        print("mel_spec_loss: ", mel_spec_loss)
        return loss, gt_messages_input

    def loss_timbre(self, watermarked_signal, watermarked_signal_with_noise, gt_messages_input, flag_index, index_flag_length):
        MSEloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        gt_messages = gt_messages_input.clone().detach()
        watermarked_signal_message = self.model.test_forward(watermarked_signal).to(self._device).squeeze()
        target_message = self.target_message_true * 2 - 1
        noise_messages = self.model.test_forward(watermarked_signal_with_noise).to(self._device).squeeze()
        
        # 判断watermark不一样的地方
        target_message = self.process_target_message(flag_index, target_message, gt_messages)

        mel_spec_loss = self.computer_melspec_loss(watermarked_signal, watermarked_signal_with_noise, device=self._device)
        wav_loss = l1loss(watermarked_signal_with_noise, watermarked_signal)
        message_loss = MSEloss(noise_messages, target_message)
        loss = wav_loss + message_loss + mel_spec_loss
        print("gt_messages:, ", gt_messages)
        print("noise_messages: ", noise_messages)
        print("target_message", target_message)
        print("wav_loss: ", wav_loss)
        print("message_loss: ", message_loss)
        print("mel_spec_loss:", mel_spec_loss)
        return loss, gt_messages

    def bwacc_audioseal(self, signal):
        result, msg_decoded = self.model.detect_watermark(signal)
        if self.on_bitstring:
            if msg_decoded is None:
                return torch.zeros(1)
            else:
                bitacc = 1 - torch.sum(torch.abs(self.target_message_true - msg_decoded)) / self.target_message_true.numel()
                return bitacc
        else:
            return result

    def bwacc_timbre(self, signal):  # signal is tensor on gpu
        payload = self.model.test_forward(signal).squeeze()  # signal: [1,1,80000]
        target_message = self.target_message_true.detach()
        target_message = target_message * 2 - 1
        payload = payload.to(self._device)
        bitacc = (payload >= 0).eq(target_message >= 0).sum().float() / target_message.numel()
        return bitacc

def get_tensor_pert(watermark_signal):
    scale = 0.001 # range 0 to 1, perturbation = scale * watermark_signal, 等比缩小
    perturbation = scale * watermark_signal
    perturbation = torch.Tensor(perturbation)
    return perturbation

def whitebox_attack_timbre(detector, watermarked_signal, index_flag, args):
    start_time = time.time()
    bwacc = detector.bwacc(watermarked_signal)
    gt_message_input = detector.model.test_forward(watermarked_signal).to(detector._device).squeeze()
    index_flag_deepcopy = copy.deepcopy(index_flag)
    index_flag_length = len(index_flag_deepcopy)
    best_bwacc = bwacc
    best_adv_signal = watermarked_signal
    tensor_pert = get_tensor_pert(watermarked_signal)
    tensor_pert.requires_grad = True
    # Freeze detector and watermarked_signal
    watermarked_signal.requires_grad = False
    # Define optimizer
    optimizer = optim.Adam([tensor_pert], lr=args.lr)
    num = 0
    # Projected Gradient Descent
    for _ in range(args.iter):
        detector.model.train()
        optimizer.zero_grad()
        watermarked_signal_with_noise = watermarked_signal + tensor_pert
        loss, gt_message_input = detector.get_loss(watermarked_signal, watermarked_signal_with_noise, gt_message_input, index_flag, index_flag_length)
        bwacc = detector.bwacc(watermarked_signal_with_noise)
        snr = 10 * torch.log10(torch.mean(watermarked_signal ** 2) / torch.mean(tensor_pert ** 2))
        print(f'Loss: {loss.item():.3f}, BWACC: {bwacc:.3f}, SNR: {snr:.1f}')
        loss.backward()
        optimizer.step()
        detector.model.eval()
        with torch.no_grad():
            num = num + 1
            print("iter_num: ", num)
            watermarked_signal_with_noise = watermarked_signal + tensor_pert
            bwacc = detector.bwacc(watermarked_signal_with_noise)
            gt_message_real = detector.model.test_forward(watermarked_signal).to(detector._device).squeeze()
            noise_message = detector.model.test_forward(watermarked_signal_with_noise).squeeze()
            snr = 10 * torch.log10(torch.mean(watermarked_signal ** 2) / torch.mean(tensor_pert ** 2))
            if bwacc > best_bwacc:
                best_bwacc = bwacc
            if index_flag:
                for index in index_flag:
                    if gt_message_real[index] < 0:
                        if noise_message[index] > 0.97577 and noise_message[index] < 1.03801:
                            gt_message_input[index] = noise_message[index]
                            index_flag.remove(index)
                    elif gt_message_real[index] >= 0:
                        if noise_message[index] < -0.96002 and noise_message[index] > -1.03475:
                            gt_message_input[index] = noise_message[index]
                            index_flag.remove(index)

            cnt = len(index_flag_deepcopy)
            if best_bwacc > args.tau:
                for index in index_flag_deepcopy:
                    if gt_message_real[index] < 0:
                        if noise_message[index] > 0.97577 and noise_message[index] < 1.03801:
                            cnt -= 1
                    elif gt_message_real[index] >= 0:
                        if noise_message[index] < -0.96002 and noise_message[index] > -1.03475:
                            cnt -= 1
                if cnt == 0:
                    best_adv_signal = watermarked_signal_with_noise
                    break
    if best_bwacc <= args.tau:
        print(f'Attack failed, the best bwacc is {best_bwacc}')
    print(f'Attack time: {time.time() - start_time}')
    return best_adv_signal, num, gt_message_input

def whitebox_attack_audioseal(detector, watermarked_signal, index_flag, args):
    _, gt_message_input = detector.model(watermarked_signal)
    gt_message_input = gt_message_input.squeeze()
    index_flag_deepcopy = copy.deepcopy(index_flag)
    index_flag_length = len(index_flag_deepcopy)
    start_time = time.time()
    bwacc = detector.bwacc(watermarked_signal)
    best_bwacc = bwacc
    best_adv_signal = watermarked_signal
    tensor_pert = get_tensor_pert(watermarked_signal)
    tensor_pert.requires_grad = True
    # Freeze detector and watermarked_signal
    watermarked_signal.requires_grad = False
    # Define optimizer
    optimizer = optim.Adam([tensor_pert], lr=args.lr)
    num = 0
    # Projected Gradient Descent
    for _ in range(args.iter):
        detector.model.train()
        optimizer.zero_grad()
        watermarked_signal_with_noise = watermarked_signal + tensor_pert
        loss, gt_message_input = detector.get_loss(watermarked_signal, watermarked_signal_with_noise, gt_message_input, index_flag, index_flag_length)
        bwacc = detector.bwacc(watermarked_signal_with_noise)
        snr = 10 * torch.log10(torch.mean(watermarked_signal ** 2) / torch.mean(tensor_pert ** 2))
        print(f'Loss: {loss.item():.3f}, BWACC: {bwacc:.3f}, SNR: {snr:.3f}')
        loss.backward()
        optimizer.step()
        # Evaluation
        detector.model.eval()
        with torch.no_grad():
            num = num + 1
            watermarked_signal_with_noise = watermarked_signal + tensor_pert
            _, watermarked_signal_message = detector.model.detect_watermark(watermarked_signal)
            watermarked_signal_message = watermarked_signal_message.squeeze()
            _ , noise_message = detector.model(watermarked_signal_with_noise)
            noise_message = noise_message.squeeze()
            bwacc = detector.bwacc(watermarked_signal_with_noise)
            snr = 10 * torch.log10(torch.mean(watermarked_signal ** 2) / torch.mean(tensor_pert ** 2))
            if bwacc > best_bwacc:
                best_bwacc = bwacc
            if index_flag:
                for index in index_flag:
                    if watermarked_signal_message[index] < 0.5:
                        if noise_message[index] > 0.78981:
                            gt_message_input[index] = noise_message[index]
                            index_flag.remove(index)
                            print("进入1")
                            print(index_flag)
                    elif watermarked_signal_message[index] >= 0.5:
                        if noise_message[index] < 0.21656:
                            gt_message_input[index] = noise_message[index]
                            index_flag.remove(index)
                            print("进入0")
                            print(index_flag)
            cnt = len(index_flag)
            if best_bwacc > args.tau:
                print("cnt: ", cnt)
                if cnt <= 0:
                    best_adv_signal = watermarked_signal_with_noise
                    break
            
    if best_bwacc <= args.tau:
        print(f'Attack failed, the best bwacc is {best_bwacc}')
    print(f'Attack time: {time.time() - start_time}')
    return best_adv_signal, num, gt_message_input

def decode_audio_files_perturb_whitebox(model, output_dir, args, device):
    watermarked_files = os.listdir(os.path.join(output_dir, 'watermark_timbre_librispeech_testing_262'))
    progress_bar = tqdm(enumerate(watermarked_files), desc="Decoding Watermarks under whitebox attack")
    save_path = os.path.join(output_dir, args.whitebox_folder)
    os.makedirs(save_path, exist_ok=True)
    visqol = api_visqol()
    for file_num, watermarked_file in progress_bar:
        filename_without_ext = os.path.splitext(watermarked_file)[0]
        idx = filename_without_ext.split("_", 1)[-1]
        waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'watermark_timbre_librispeech_testing_262', watermarked_file))
        print("----------------watermarked_file: ", watermarked_file, "--------------------------")

        '''waveform.shape = [1, 80000]'''
        waveform = waveform.to(device=device)
        waveform = waveform.unsqueeze(0)

        original_payload_str = watermarked_file.split('_')[0]
        original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.float32, device=device)

        attacked_message_str = "1100110011001100"
        attacked_message = torch.tensor(list(map(int, attacked_message_str)), dtype=torch.float32, device=device)
        
        message_length = len(attacked_message_str)
        
        attack_bitstring = True

        flag = -1
        for index in range(0, message_length):
            if attacked_message[index] != original_payload[index]:
                flag = index
                break

        detector = WatermarkDetectorWrapper(model, original_payload, attacked_message, attack_bitstring, args.model, args.tau, device)

        if args.model == "audioseal":
            flag_index = []
            for index in range(0, message_length):
                if attacked_message.squeeze()[index] != original_payload.squeeze()[index]:
                    flag_index.append(index)
            print("flag_idex: ", flag_index)
            adv_signal, num_message, gt_message_input = whitebox_attack_audioseal(detector, waveform, flag_index, args)
            tensor_pert = adv_signal - waveform
            snr = 10 * torch.log10(torch.mean(waveform ** 2) / torch.mean(tensor_pert ** 2))
        elif args.model == "timbre":
            flag_index = []
            for index in range(0, message_length):
                if attacked_message.squeeze()[index] != original_payload.squeeze()[index]:
                    flag_index.append(index)
                    # print(index)
            print("flag_idex: ", flag_index)
            adv_signal, num_message, gt_message_input = whitebox_attack_timbre(detector, waveform, flag_index, args)
            tensor_pert = adv_signal - waveform
            snr = 10 * torch.log10(torch.mean(waveform ** 2) / torch.mean(tensor_pert ** 2))
        
        '''save to log file'''
        filename = os.path.join(save_path, f'whitebox.csv')
        log = open(filename, 'a' if os.path.exists(filename) else 'w')
        log.write('idx, query, acc, snr, visqol\n')
        acc = detector.bwacc(adv_signal)
        snr = 10 * torch.log10(torch.sum(waveform ** 2) / torch.sum((adv_signal - waveform) ** 2))
        visqol_score = visqol.Measure(np.array(waveform.squeeze().detach().cpu(), dtype=np.float64),
                                      np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)).moslqo
        log.write(f'{idx}, {num_message}, {acc}, {snr}, {visqol_score}\n')
        torchaudio.save(os.path.join(save_path,
                                         f"{idx}_tau{args.tau}_query_message{num_message}_snr{snr:.3f}_acc{acc:.3f}_visqol{visqol_score}.wav"),
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
        output_dir = 'attack_audioseal'
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
        output_dir = 'attack_timbre'
        os.makedirs(output_dir, exist_ok=True)

    decode_audio_files_perturb_whitebox(model, output_dir, args, device)


if __name__ == "__main__":
    main()
