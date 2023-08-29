import numpy as np
import librosa
import os
from tqdm import tqdm
from iso_mvdr_metric import ISO_MVDR
import glob
import argparse

def rtf_error(y, y_hat, n_fft=2048):
    '''
    y: 8, n_samples
    y_hat: 8, n_samples
    '''
    n_channels, n_samples = y.shape
    y_stft = librosa.stft(y, n_fft=n_fft, hop_length=int(n_fft//4), win_length=n_fft, window='hann', center=True, dtype=None, pad_mode='constant')
    y_hat_stft = librosa.stft(y_hat, n_fft=n_fft, hop_length=int(n_fft//4), win_length=n_fft, window='hann', center=True, dtype=None, pad_mode='constant') # 8, F, T
    
    # normalize before svd:
    y_stft = y_stft - y_stft.mean(-1)[..., None]
    y_hat_stft = y_hat_stft - y_hat_stft.mean(-1)[..., None]
    
    n_channels, n_freqs, n_frames = y_stft.shape
    rtf_error_per_freq = []
    cosine_sim_per_freq = []
    
    for f in range(n_freqs):
        y_U, y_S, _ = np.linalg.svd(y_stft[:, f, :])
        y_hat_U, y_hat_S, _ = np.linalg.svd(y_hat_stft[:, f, :])
        
        y_rtf = y_U[:, 0]
        y_hat_rtf = y_hat_U[:, 0]
        
        y_rtf_norm = np.sqrt((y_rtf * np.conjugate(y_rtf)).sum() + 1e-12)
        y_hat_rtf_norm = np.sqrt((y_hat_rtf * np.conjugate(y_hat_rtf)).sum() + 1e-12)
        
        current_rtf_error = (y_rtf * np.conjugate(y_hat_rtf)).sum() / (y_rtf_norm * y_hat_rtf_norm + 1e-12)
        cosine_sim = np.abs(current_rtf_error)
        current_rtf_error = np.arccos(current_rtf_error.real)
        
        rtf_error_per_freq.append(current_rtf_error)
        cosine_sim_per_freq.append(cosine_sim)
        
    error = np.array(rtf_error_per_freq).mean()
    # cosine_score = np.array(cosine_sim_per_freq).mean()
    return error

rtf_errors = []
spatial_similariy = []

class SpatialMetric:
    def __init__(self) -> None:
        self.beamformer = ISO_MVDR()
    def rtf_error(self, y, y_hat, n_fft=2048):
        '''
        y: 8, n_samples
        y_hat: 8, n_samples
        '''
        return rtf_error(y, y_hat, n_fft)
    
    def spatial_similarity(self, y, y_hat):
        '''
        y: 8, n_samples
        y_hat: 8, n_samples
        '''
        return self.beamformer.cosine_similarity(y, y_hat)
    
    
model = SpatialMetric()
x = np.ones((8, 16000))
x_hat = np.ones((8, 16000))

a = model.rtf_error(x, x_hat)
b = model.spatial_similarity(x, x_hat)

print(a, b)