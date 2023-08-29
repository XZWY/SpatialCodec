from tqdm import tqdm
from models import Quantizer
from spatial_model_subband import SpatialEncoder, SpatialDecoder
import torch
import soundfile as sf
import os
import shutil
import json
import torch.nn.functional as F
import numpy as np
from frequency_codec import Encoder, Generator
import librosa
import glob
import argparse

device='cuda:0'

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

with open('config_spatialcodec.json') as f:
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

# ckpt_dir = '/media/alan/新加卷/SpatialCodec/SpatialCodec/ckpts_outport/ckpts_outport_final/spatial_codec_cov_suband_coding_full_conv_6kbps_deepfilter/g_00760000'
# ckpt_dir_ref = '/media/alan/新加卷/SpatialCodec/SpatialCodec/ckpts_outport/ckpts_outport_final/ref_reverb_clean_subband_codec6kbps_snr/g_01210000'
# input_dir = '/media/alan/新加卷/SpatialCodec/final_inference_samples/reverb_clean_mc'
# output_dir = '/media/alan/新加卷/SpatialCodec/final_inference_samples/spatial_codec_est_clean_snr'
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--ref_ckpt_dir', type=str, required=True)
parser.add_argument('--spatial_ckpt_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

quantizer = Quantizer(h, n_dim=256*6).to(device)
encoder = SpatialEncoder().to(device)
decoder = SpatialDecoder().to(device)

ref_quantizer = Quantizer(h, n_dim=256*6).to(device)
ref_encoder = Encoder().to(device)
ref_decoder = Generator().to(device)

ckpt = torch.load(args.spatial_ckpt_dir, map_location=device)
encoder.load_state_dict(ckpt['encoder'], strict=False)
decoder.load_state_dict(ckpt['decoder'], strict=False)
quantizer.load_state_dict(ckpt['quantizer'], strict=False)

ckpt = torch.load(args.ref_ckpt_dir, map_location=device)
ref_encoder.load_state_dict(ckpt['encoder'], strict=False)
ref_decoder.load_state_dict(ckpt['generator'], strict=False)
ref_quantizer.load_state_dict(ckpt['quantizer'], strict=False)

encoder.eval()
decoder.eval()
quantizer.eval()

ref_quantizer.eval()
ref_encoder.eval()
ref_decoder.eval()

input_file_names = glob.glob(args.input_dir+'/*.wav')

with torch.no_grad():
    for j in tqdm(range(len(input_file_names))):
        # -------------------------------------
        filename = input_file_names[j].split('/')[-1]
        
        reverb_clean, sr = librosa.load(input_file_names[j], sr=16000, mono=False)
        reverb_clean = torch.tensor(reverb_clean)
        reverb_clean = torch.autograd.Variable(reverb_clean.to(device, non_blocking=True))
        
        _, _, cov = get_cov_matrix(reverb_clean)
        
        cov = cov.unsqueeze(0)
        reverb_clean = reverb_clean.unsqueeze(0)

        bs, _, n_freqs, n_frames = cov.shape # bs, 72, F, T
        
        ref_emb = ref_encoder(reverb_clean[:, 3, ...].unsqueeze(1))
        q_ref, _, _ = ref_quantizer(ref_emb)
        ref_g = ref_decoder(q_ref) # bs, 1, n_samples
        
        ref_g_stft = torch.stft(ref_g.reshape(bs, -1), 640, hop_length=320, win_length=640, window=torch.hann_window(640).to(reverb_clean.device),
                        center=True, pad_mode='constant', normalized=False, onesided=True, return_complex=True) # bs, F, T
        
        # encoder input: # bs, 72, F, T
        cov = cov.permute(0,1,3,2)
        c_emb = encoder(cov) # bs, 9*512, n_frames

        q_rtf, loss_q_rtf, _ = quantizer(c_emb) # bs, 22*512, n_frames

        # decode
        rtf_g = decoder(q_rtf) # bs, 7, n_freqs, 5, n_frames, 2
        
        # decide to use est ref (real)
        ref_stft = ref_g_stft.unsqueeze(1)
        ref_stft_df = F.unfold(ref_stft, kernel_size=(3, 9), padding=(1,4)).reshape(bs, 1, 3,9,n_freqs, n_frames) # bs, 7, 3, 9, n_freqs, n_frames
        rtf_g = torch.view_as_complex(rtf_g).contiguous() # [2, 7, 3, 9, 321, 100]
        est_stfts = (rtf_g * ref_stft_df).sum(2).sum(2) # bs, 7, n_freqs, n_framesest_stfts
        # est_stfts = rtf_g * ref_stft
        est_stfts = est_stfts.reshape(bs*7, n_freqs, n_frames)
        est_reverb_clean = torch.istft(est_stfts, 640, 320, 640, window=torch.hann_window(640).to(est_stfts.device)) # bs*7, T
        est_reverb_clean = est_reverb_clean.reshape(bs, 7, -1)        
        valid_len = np.minimum(est_reverb_clean.shape[-1], ref_g.shape[-1])
        est_reverb_clean_all = torch.cat([est_reverb_clean[:, :3, :valid_len], ref_g, est_reverb_clean[:, 3:, :valid_len]], dim=1) # bs, 8, n_samples

        est_reverb_clean_all = est_reverb_clean_all[0].T.detach().cpu().numpy()
        # sf.write('/data2/v_weiyangxu/final_inference_samples/spatial_codec_est_clean_snr/sample_'+str(j)+'.wav', est_reverb_clean_all, 16000)
        
        sf.write(os.path.join(args.output_dir, filename), est_reverb_clean_all, 16000)
