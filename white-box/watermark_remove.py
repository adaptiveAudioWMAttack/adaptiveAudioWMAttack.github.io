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
    parser.add_argument("--iter", type=int, default=10000, help="Number of iterations for the attack")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the attack")
    parser.add_argument("--whitebox_folder", type=str, default="whitebox_debug", help="Folder to save the whitebox attack results")
    parser.add_argument("--tau", type=float, default=0.5, help="Threshold for the detector")
    parser.add_argument("--attack_bitstring", action="store_true", default=False, help="If set, perturb the bitstring instead of the detection probability")
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

    def loss_audioseal(self, watermarked_signal, watermarked_signal_with_noise, gt_message):
        results, messages = self.model(watermarked_signal_with_noise)
        MSEloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        BCEloss = nn.BCELoss()
        gt_message = torch.full_like(gt_message, 0.5)
        noise_class = results[:, 1, :]
        gt_class = torch.full_like(noise_class, 0.0001)
        message_loss = MSEloss(messages.squeeze(), gt_message)
        position_loss = BCEloss(gt_class, noise_class)
        wav_loss = l1loss(watermarked_signal_with_noise, watermarked_signal)
        mel_spec_loss = self.computer_melspec_loss(watermarked_signal, watermarked_signal_with_noise, device=self._device)
        loss = wav_loss + mel_spec_loss + position_loss + 16 * message_loss
        print("messages: ", messages)
        print("gt_message: ", gt_message)
        print("wav_loss: ", wav_loss)
        print("mel_spec_loss: ", mel_spec_loss)
        print("position_loss: ", position_loss)
        print("message_loss:", message_loss)
        return loss, gt_message
    
    def loss_timbre(self, watermarked_signal, watermarked_signal_with_noise, gt_message):
        MSEloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        message = self.model.test_forward(watermarked_signal_with_noise).to(self._device).squeeze()
        wav_loss = l1loss(watermarked_signal_with_noise, watermarked_signal)
        mel_spec_loss = self.computer_melspec_loss(watermarked_signal, watermarked_signal_with_noise, device=self._device)
        gt_message = torch.full_like(gt_message, 0)
        message_loss = MSEloss(message, gt_message)
        loss = wav_loss + mel_spec_loss + 10 * message_loss
        print("message: ", message)
        print("gt_message: ", gt_message)
        print("wav_loss: ", wav_loss)
        print("mel_spec_loss: ", mel_spec_loss)
        print("message_loss: ", message_loss)
        return loss, gt_message  

    def bwacc_audioseal(self, signal):
        result, msg_decoded = self.model.detect_watermark(signal)
        msg_decoded = msg_decoded.float().squeeze()
        message = 1-self.message
        if msg_decoded is None:
            return torch.zeros(1)
        else: 
            bitacc = 1 - torch.sum(torch.abs(message - msg_decoded)) / message.numel()
            return bitacc

    def bwacc_timbre(self, signal):  #signal is tensor on gpu
        payload = self.model.test_forward(signal).squeeze() # signal: [1,1,80000]
        message = (1 - self.message) * 2 - 1
        bitacc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return bitacc

def get_tensor_pert(watermark_signal):
    scale = 0.01 # range 0 to 1, perturbation = scale * watermark_signal, 等比缩小
    perturbation = scale * watermark_signal
    perturbation = torch.Tensor(perturbation)
    return perturbation

def whitebox_attack(detector, watermarked_signal, args):
    start_time = time.time()
    bwacc = detector.bwacc(watermarked_signal)
    best_bwacc = bwacc
    best_adv_signal = watermarked_signal
    if args.model == "audioseal":
        gt_message = detector.message.clone().detach()
        index_flag = list(range(16))
        percentage = 16
    if args.model == "timbre":
        gt_message = detector.message.clone().detach()
        gt_message = gt_message * 2 - 1
        index_flag = list(range(16))
        percentage = 16
    # Initialize tensor_pert
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
        loss, gt_message = detector.get_loss(watermarked_signal, watermarked_signal_with_noise, gt_message)
        bwacc = detector.bwacc(watermarked_signal_with_noise)
        snr = 10 * torch.log10(torch.mean(watermarked_signal ** 2) / torch.mean(tensor_pert ** 2))
        print(f'Loss: {loss.item():.5f}, BWACC: {bwacc:.3f}, SNR: {snr:.3f}')
        loss.backward()
        optimizer.step()
        detector.model.eval()
        with torch.no_grad():
            num += 1
            print("iter: ", num)
            watermarked_signal_with_noise = watermarked_signal + tensor_pert 
            bwacc = detector.bwacc(watermarked_signal_with_noise)
            snr = 10*torch.log10(torch.mean(watermarked_signal**2)/torch.mean(tensor_pert**2))
            if args.model == "audioseal":
                _, watermarked_signal_message = detector.model.detect_watermark(watermarked_signal)
                watermarked_signal_message = watermarked_signal_message.squeeze()
                _ , noise_message = detector.model(watermarked_signal_with_noise)
                noise_message = noise_message.squeeze()
                cnt = percentage
            if args.model == "timbre":
                noise_message = detector.model.test_forward(watermarked_signal_with_noise).squeeze()
                cnt = percentage
            if bwacc < best_bwacc:
                best_bwacc = bwacc
                best_adv_signal = watermarked_signal_with_noise
            if args.model == "audioseal":
                if bwacc < args.tau:
                    for index in index_flag:
                        if noise_message[index] > 0.49177 and noise_message[index] < 0.51039:
                            cnt -= 1
                if cnt <= 0:
                    best_adv_signal = watermarked_signal_with_noise
                    break
            print("cnt: ", cnt)
            if args.model == "timbre":
                if bwacc < args.tau:
                    for index in index_flag:
                        if noise_message[index] > -0.05589 and noise_message[index] < 0.07426:
                            cnt -= 1
                if cnt <= 0:
                    best_adv_signal = watermarked_signal_with_noise
                    break
    if best_bwacc > args.tau:
        print(f'Attack failed, the best bwacc is {best_bwacc}')
    print(f'Attack time: {time.time() - start_time}')
    return best_adv_signal, num


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

        '''waveform.shape = [1, 80000]'''
        waveform = waveform.to(device=device)
        waveform = waveform.unsqueeze(0)

        original_payload_str = watermarked_file.split('_')[0]
        original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.float32, device=device)
        original_payload = 1 - original_payload

        attack_bitstring = True

        detector = WatermarkDetectorWrapper(model, original_payload, args.attack_bitstring, args.model, args.tau, device)
        adv_signal, num = whitebox_attack(detector, waveform, args)

        '''save to log file'''
        filename=os.path.join(save_path, f'whitebox.csv')
        log = open(filename, 'a' if os.path.exists(filename) else 'w')
        log.write('idx, query, acc, snr, visqol\n')
        acc = detector.bwacc(adv_signal)
        snr = 10*torch.log10(torch.sum(waveform**2)/torch.sum((adv_signal - waveform)**2))
        visqol_score = visqol.Measure(np.array(waveform.squeeze().detach().cpu(), dtype=np.float64), np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)).moslqo
        print(f'idx: {idx}, query: {num}, acc: {acc:.3f}, snr: {snr:.1f}, visqol: {visqol_score:.3f}')
        log.write(f'{idx}, {num}, {acc}, {snr}, {visqol_score}\n')
        torchaudio.save(os.path.join(save_path, 
            f"{idx}_tau{args.tau}_query{num}_snr{snr:.1f}_acc{acc:.3f}_visqol{visqol_score:.3f}.wav"),
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