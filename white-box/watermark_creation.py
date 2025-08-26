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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with audioseal")
    parser.add_argument("--gpu", type=int, default=1, help="GPU device index to use")
    parser.add_argument("--iter", type=int, default=100000, help="Number of iterations for the attack")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the attack")
    parser.add_argument("--whitebox_folder", type=str, default="whitebox_debug", help="Folder to save the whitebox attack results")
    parser.add_argument("--tau", type=float, default=0.95, help="Threshold for the detector")
    parser.add_argument("--attack_bitstring", action="store_true", default=True, help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--model", type=str, default='audioseal', choices=['audioseal', 'timbre'], help="Model to be attacked")

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
    
    def loss_audioseal(self, signal, watermark_signal, gt_message):
        mel_spec_loss = self.computer_melspec_loss(signal, watermark_signal, device=self._device)
        results, messages = self.model(watermark_signal)
        MSEloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        BCEloss = nn.BCELoss()
        wav_loss = l1loss(signal, watermark_signal)
        message_loss = MSEloss(messages.squeeze(), gt_message)
        class_1 = results[:, 1, :]
        gt_class = torch.full_like(class_1, 0.9999)
        m = nn.Sigmoid()
        noise_class = m(class_1)
        position_loss = BCEloss(gt_class, noise_class)
        loss = wav_loss + mel_spec_loss + position_loss + 16 * message_loss
        print("==============================================")
        print("messages.squeeze(): ", messages.squeeze())
        print("gt_message", gt_message)
        print("wav_loss: ", wav_loss)
        print("mel_spec_loss: ", mel_spec_loss)
        print("position_loss: ", position_loss)
        print("message_loss:", message_loss)
        return loss, gt_message
        
    def loss_timbre(self, signal, watermark_signal, gt_message):
        MSEloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        mel_spec_loss = self.computer_melspec_loss(signal, watermark_signal, device=self._device)
        target_message = self.model.test_forward(watermark_signal).squeeze()
        wav_loss = l1loss(signal, watermark_signal)
        message_loss = MSEloss(target_message, gt_message)
        loss = wav_loss + mel_spec_loss + message_loss
        print("gt_message: ", gt_message)
        print("target_message: ", target_message)
        print("wav_loss: ", wav_loss)
        print("mel_spec_loss: ", mel_spec_loss)
        print("message_loss: ", message_loss)
        print("loss: ", loss)
        return loss, gt_message

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

    def bwacc_timbre(self, signal):  #signal is tensor on gpu
        payload = self.model.test_forward(signal).squeeze().to(self._device)
        message = self.message * 2 - 1
        bitacc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return bitacc

def get_tensor_pert(watermark_signal):
    scale = 0.001 # range 0 to 1, perturbation = scale * watermark_signal, 等比缩小
    perturbation = scale * watermark_signal
    perturbation = torch.Tensor(perturbation)
    return perturbation

def whitebox_attack(detector, signal, args):
    start_time = time.time()
    bwacc = detector.bwacc(signal)
    best_bwacc = bwacc
    best_adv_signal = signal
    tensor_pert = get_tensor_pert(signal)
    for param in detector.model.parameters():
        param.requires_grad = False
    if args.model == "audioseal":
        gt_message = detector.message.clone().detach()
        index_flag = list(range(16))
        percentage = int(16.0 - bwacc * 16.0)
        if percentage < 4:
            percentage = 4
        if percentage > 8:
            percentage = 8
    if args.model == "timbre":
        gt_message = detector.message.squeeze().clone().detach()
        gt_message = gt_message * 2 - 1
        index_flag = list(range(16))
        percentage = 16
    tensor_pert.requires_grad = True
    # Freeze detector and signal
    signal.requires_grad = False
    # Define optimizer
    optimizer = optim.Adam([tensor_pert], lr=args.lr)
    num = 0
    # Projected Gradient Descent
    for _ in range(args.iter):
        detector.model.train()
        optimizer.zero_grad()
        watermarked_signal = signal + tensor_pert
        loss, gt_message = detector.get_loss(signal, watermarked_signal, gt_message)
        bwacc = detector.bwacc(watermarked_signal)
        snr = 10*torch.log10(torch.mean(signal**2)/torch.mean(tensor_pert**2))
        print(f'Loss: {loss.item():.3f}, BWACC: {bwacc:.3f}, SNR: {snr:.1f}')
        loss.backward()
        optimizer.step()
        detector.model.eval()
        with torch.no_grad():
            num = num + 1
            print("iter: ", num)
            watermarked_signal = signal + tensor_pert 
            bwacc = detector.bwacc(watermarked_signal)
            snr = 10*torch.log10(torch.mean(signal**2)/torch.mean(tensor_pert**2))
            if bwacc > best_bwacc:
                best_bwacc = bwacc
                best_adv_signal = watermarked_signal
            if args.model == "audioseal":
                _ , noise_message = detector.model(watermarked_signal)
                noise_message = noise_message.squeeze()
                cnt = percentage
            if args.model == "timbre":
                noise_message = detector.model.test_forward(watermarked_signal).squeeze()
                print("noise_message: ", noise_message)
                if torch.isnan(noise_message).any():
                    print("Warning: noise_message contains NaN values. Skipping this iteration.")
                    num = -1
                    break
                cnt = percentage
            if args.model == "audioseal":
                for index in index_flag:
                    if index == -1:
                        continue
                    else:
                        if noise_message[index] > 0.78981:
                             gt_message[index] = noise_message[index]
                             index_flag[index] = -1
                             best_adv_signal = watermarked_signal
                        elif noise_message[index] < 0.21656:
                            gt_message[index] = noise_message[index]
                            index_flag[index] = -1
                            best_adv_signal = watermarked_signal
                print("index_flag: ", index_flag)
                for j in index_flag:
                    if j == -1:
                        cnt -= 1
                if cnt <= 0:
                    break
            if args.model == "timbre":
                if best_bwacc > args.tau:
                    for index in index_flag:
                        if noise_message[index] > 0.97577 and noise_message[index] < 1.03801:
                            cnt -= 1
                        elif noise_message[index] < -0.96002 and noise_message[index] > -1.03475:
                            cnt -= 1
                    if cnt <= 0:
                        best_adv_signal = watermarked_signal
                        break           
    if best_bwacc <= args.tau:
        print(f'Attack failed, the best bwacc is {best_bwacc}')
    print(f'Attack time: {time.time() - start_time}')
    return best_adv_signal, num


def decode_audio_files_perturb_whitebox(model, output_dir, args, device):
    watermarked_files = os.listdir(os.path.join(output_dir, 'timbre_librispeech_testing_262'))
    progress_bar = tqdm(enumerate(watermarked_files), desc="Decoding Watermarks under whitebox attack")
    save_path = os.path.join(output_dir, args.whitebox_folder)
    os.makedirs(save_path, exist_ok=True)   
    visqol = api_visqol()
    for file_num, watermarked_file in progress_bar:
        idx = os.path.splitext(watermarked_file)[0]
        waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'timbre_librispeech_testing_262', watermarked_file))
        print("watermarked_file: ", watermarked_file)

        '''waveform.shape = [1, 80000]'''
        waveform = waveform.to(device=device)
        waveform = waveform.unsqueeze(0)

        original_payload_str = "1100110011001100"
        original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.float32, device=device)

        attack_bitstring = True

        detector = WatermarkDetectorWrapper(model, original_payload, args.attack_bitstring, args.model, args.tau, device)
        adv_signal, num = whitebox_attack(detector, waveform, args)

        if num == -1:
            continue

        '''save to log file'''
        filename=os.path.join(save_path, f'whitebox.csv')
        log = open(filename, 'a' if os.path.exists(filename) else 'w')
        log.write('idx, query, acc, snr, visqol\n')
        acc = detector.bwacc(adv_signal)
        snr = 10*torch.log10(torch.sum(waveform**2)/torch.sum((adv_signal - waveform)**2))
        visqol_score = visqol.Measure(np.array(waveform.squeeze().detach().cpu(), dtype=np.float64), np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)).moslqo
        print(f'idx: {idx}, query: {num}, acc: {acc:.3f}, snr: {snr:.1f}, visqol: {visqol_score}')
        log.write(f'{idx}, {num}, {acc}, {snr}, {visqol_score}\n')
        torchaudio.save(os.path.join(save_path,
            f"{idx}_tau{args.tau}_query{num}_snr{snr:.1f}_acc{acc:.1f}_visqol{visqol_score}.wav"),
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