from tqdm import tqdm
from models import Quantizer
from spatial_model_subband import SpatialEncoder, SpatialDecoder
import torch
import soundfile as sf
import os
import json
import torch.nn.functional as F
import numpy as np
import librosa
import glob
import argparse

device='cuda:0'

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

with open('config_mimo_e2e.json') as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

def get_cov_matrix(array_audio):
    '''
    array_audio: n_channels, n_samples
    '''
    array_stft = torch.stft(array_audio, 640, hop_length=320, win_length=640, window=torch.hann_window(640).to(device),
                    center=True, pad_mode='constant', normalized=False, onesided=True, return_complex=True) # n_channels, F, T
    ref_stft = array_stft[3].unsqueeze(0)
    n_channels, n_freqs, n_frames = array_stft.shape
    
    array_stft = array_stft.permute(1,2,0).unsqueeze(2) # F, T, 1, n_channels
    array_stft_t = array_stft.permute(0,1,3,2) # F, T, n_channels, 1
    
    cov_matrix = torch.matmul(array_stft_t, torch.conj(array_stft)) # F, T, n_channels, n_channels
    
    mask = torch.ones(n_channels, n_channels).to(array_audio.device)
    mask = (torch.triu(mask)==1)
    cov_matrix_upper = cov_matrix[:, :, mask]
    cov_matrix_upper = torch.view_as_real(cov_matrix_upper).permute(2,3,0,1).reshape(-1, n_freqs, n_frames)
    ref_stft = torch.view_as_real(ref_stft)
    
    features = torch.cat([cov_matrix_upper, ref_stft.squeeze(0).permute(2,0,1)], dim=0)
    
    return cov_matrix_upper, ref_stft, features

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--ckpt_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

quantizer = Quantizer(h, n_dim=256*6).to(device)
encoder = SpatialEncoder().to(device)
decoder = SpatialDecoder().to(device)

ckpt = torch.load(args.ckpt_dir, map_location=device)
encoder.load_state_dict(ckpt['spatial_encoder'], strict=False)
decoder.load_state_dict(ckpt['spatial_decoder'], strict=False)
quantizer.load_state_dict(ckpt['spatial_quantizer'], strict=False)

encoder.eval()
decoder.eval()
quantizer.eval()


input_file_names = glob.glob(args.input_dir+'/*.wav')
with torch.no_grad():
    for j in tqdm(range(len(input_file_names))):
        # -------------------------------------
        filename = input_file_names[j].split('/')[-1]
        mc_audio, sr = librosa.load(input_file_names[j], sr=16000, mono=False)
        mc_audio = torch.tensor(mc_audio)
        mc_audio = torch.autograd.Variable(mc_audio.to(device, non_blocking=True))
        
        _, _, cov = get_cov_matrix(mc_audio)
        
        cov = cov.unsqueeze(0)

        bs, _, n_freqs, n_frames = cov.shape # bs, 74, F, T
        # bs, F, T
        cov = cov.permute(0,1,3,2)
        c_emb = encoder(cov) # bs, 9*512, n_frames
        q, loss_q, _ = quantizer(c_emb) # bs, 22*512, n_frames
        y_g_hat = decoder(q) # bs, 8, n_samples
        
        y_g_hat = y_g_hat.reshape(bs*8, -1)
        
        y_g_hat = y_g_hat.T.detach().cpu().numpy()
        
        sf.write(os.path.join(args.output_dir, filename), y_g_hat, 16000)
