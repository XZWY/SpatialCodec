'''
super-directive mvdr for linear array, numpy version
2023-08-01 by weiyangxu
applications:
    1. angle features (better directivity / sharper beam than delay sum)
    2. evaluation for spatial cue preservations
'''

import numpy as np
import matplotlib.pyplot as plt
import librosa

class ISO_MVDR:
    def __init__(self, k=50) -> None:
        distances = np.array([0., 0.02, 0.02, 0.02, 0.14, 0.02, 0.02, 0.02])
        self.distances_acc = distances.cumsum()
        # self.angles = np.linspace(0, 2*np.pi, 36)
        self.K = k # number of beamformers
        self.angles = np.array([np.arccos(1 - 2 * k / (self.K - 1)) for k in range(self.K)])
        
        diagonal_loading_lambda = 1e-5
        self.n_fft = 2048
        self.n_freqs = int(self.n_fft//2+1)
        self.n_angles = len(self.angles)
        self.n_channels = 8
        self.C = 343.
        self.fs = 16000
        
        # initialize steering vectors
        self.steer_vecs = np.zeros((self.n_angles, self.n_freqs, self.n_channels), dtype=np.complex64)
        for i in range(self.n_angles):
            for j in range(self.n_freqs):
                for k in range(self.n_channels):
                    phase = - 2 * np.pi * self.distances_acc[k] * np.cos(self.angles[i]) / self.C * j / self.n_fft * self.fs
                    self.steer_vecs[i, j, k] = np.exp(1j * phase)
        
        # initialize isotropic diffused SCMs (sinc(2 * f * d / c) where sinc(x) = sin(pi * x) / (pi * x)) 
        self.phi_nn = np.zeros((self.n_freqs, self.n_channels, self.n_channels), dtype=np.complex64) # F, C, C
        for i in range(self.n_freqs):
            for j in range(self.n_channels):
                for k in range(self.n_channels):
                    self.phi_nn[i, j, k] = np.sinc(2 * np.abs(self.distances_acc[j] - self.distances_acc[k]) * i * self.fs / (self.C * self.n_fft))
        
        # diagonal loading (o.w. illposed for some freqs)
        self.phi_nn += (diagonal_loading_lambda * np.eye(self.n_channels))[np.newaxis, ...]

        # beamforming weights, conjugate: n_angles, n_freqs, n_channels
        self.beamformer_weights_H = np.ones((self.n_angles, self.n_freqs,  self.n_channels), dtype=np.complex64)
        for i in range(self.n_freqs):
            for j in range(self.n_angles):
                d = self.steer_vecs[j, i, :, np.newaxis]
                phi_inv_d = np.linalg.pinv(self.phi_nn[i]) @ d
                d_H = np.conjugate(d.T)
                self.beamformer_weights_H[j, i] = np.conjugate((phi_inv_d / (d_H @ phi_inv_d))[:,0])
    
    def get_steer_vec(self, angle):
        '''
        angle in degrees, 0 degree corresponding to left parallel to the array, increases clockwisely to 180
        '''
        angle = angle / 180 * np.pi
        phase = [[- 2 * np.pi * self.distances_acc[k] * np.cos(angle) / self.C * j / self.n_fft * self.fs for k in range(self.n_channels)] for j in range(self.n_freqs)]
        steer_vec = np.exp(1j * np.array(phase)) # n_freqs, n_channels
        # steer_vec = steer_vec / steer_vec[3]
        return steer_vec
    
    def beamform(self, input, angle):
        input_stft = self.stft(input) # n_channels, n_freqs, n_frames
        stft = np.transpose(input_stft, [1,2,0])[..., np.newaxis] # n_freqs, n_frames, n_channels, 1
        
        steer_vec = self.get_steer_vec(angle) # n_freqs, 8
        beamformer_weights_H = np.ones((self.n_freqs,  self.n_channels), dtype=np.complex64)
        for i in range(self.n_freqs):
            d = steer_vec[i, :, np.newaxis]
            phi_inv_d = np.linalg.pinv(self.phi_nn[i]) @ d
            d_H = np.conjugate(d.T)
            beamformer_weights_H[i] = np.conjugate((phi_inv_d / (d_H @ phi_inv_d))[:,0])
        
        output = beamformer_weights_H[:, np.newaxis, np.newaxis, :] @ stft # n_freqs, 1, 1 n_channels
        output = output[:, :, 0, 0] # n_freqs, n_frames
        output = self.istft(output)
        return output
    
    def stft(self, input):
        return librosa.stft(input, n_fft=self.n_fft, hop_length=int(self.n_fft // 4), win_length=self.n_fft, window='hann')

    def istft(self, input):
        return librosa.istft(input, hop_length=int(self.n_fft // 4), win_length=self.n_fft, n_fft=self.n_fft)    
    
    def get_features(self, input, squeeze_time=True):
        '''
        input: 8, n_samples
        
        output: n_angles, n_freqs, n_frames or n_angles, n_freqs
        '''
        stft = self.stft(input) # n_channels, n_freqs, n_frames
        stft = np.transpose(stft, [1,2,0])[np.newaxis, ..., np.newaxis] # 1, n_freqs, n_frames, n_channels, 1
        
        features = self.beamformer_weights_H[:, :, np.newaxis, np.newaxis, :] @ stft # n_angles, n_freqs, 1, 1, n_channels @ 1, n_freqs, n_frames, n_channels, 1
        
        features = features[:, :, :, 0, 0] # n_angles, n_freqs, n_frames
        
        if squeeze_time:
            features = np.absolute(features).mean(axis=2)

        return features # angles, n_freqs, n_frames
    
    def cosine_similarity(self, x, y):
        '''
        x: 8, n_samples
        y: n_samples
        '''
        x_features = self.get_features(x).T # n_freqs, n_angles
        y_features = self.get_features(y).T
        
        x_feature_norm = np.linalg.norm(x_features, axis=1)
        y_feature_norm = np.linalg.norm(y_features, axis=1)
        
        # cosine_sim = np.conjugate(x_features)[:, np.newaxis, :] @ y_features[:, :, np.newaxis]
        cosine_sim = x_features[:, np.newaxis, :] @ y_features[:, :, np.newaxis]
        cosine_sim = cosine_sim[:, 0, 0] / (x_feature_norm * y_feature_norm + 1e-8)
        
        return np.abs(cosine_sim).mean()
    
    def get_beam_response(self, x, freq=[1000, 3000]):
        x_features = self.get_features(x).T # n_freqs, n_angles
        f1 = int(freq[0] / self.fs * self.n_fft)
        f2 = int(freq[1] / self.fs * self.n_fft)
        
        return x_features[f1], x_features[f2]
    
    def plot_beam_responses(self, inputs, labels):
        '''
        input: 8, n_samples
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), subplot_kw={'projection': 'polar'})
        
        styles = ['-', '--', '--', '--']
        for i in range(len(inputs)):
            feature1, feature2 = self.get_beam_response(inputs[i])
            feature1 = np.abs(feature1)
            feature2 = np.abs(feature2)
            # feature1 = feature1 / np.linalg.norm(feature1)
            # feature2 = feature2 / np.linalg.norm(feature2)
            feature1 = feature1 / feature1.max()
            feature2 = feature2 / feature2.max()
            
            # for beam_id in range(self.K):
            ax1.plot(self.angles, feature1, styles[i], label=labels[i], alpha=0.7) # MAKE SURE TO USE RADIAN FOR POLAR
            ax2.plot(self.angles, feature2, styles[i], label=labels[i], alpha=0.7)
            # ax1.plot(self.angles, feature1, label=labels[i], alpha=0.5) # MAKE SURE TO USE RADIAN FOR POLAR
            # ax2.plot(self.angles, feature2, label=labels[i], alpha=0.5)
                
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        ax2.set_thetamin(0)
        ax2.set_thetamax(180)
    
        ax1.set_title('1kHz')
        ax2.set_title('3kHz')
        ax1.set_theta_zero_location('W') # make 0 degrees point up
        ax1.set_theta_direction(-1) # increase clockwise
        ax2.set_theta_zero_location('W') # make 0 degrees point up
        ax2.set_theta_direction(-1) # increase clockwise
        # ax.set_rgrids([-10, -5, 0])
        # ax.set_rgrids([-8, -6, -4, -2, 0])
        # ax.set_rgrids([-40, -30, -20, -10, 0])
        # ax1.set_rgrids([0.2, 0.4, 0.6, 0.8, 1])

        
        import matplotlib.ticker as ticker
        # plotting code here
        # frmtr = ticker.FormatStrFormatter('%1.1e')
        # ax1.yaxis.set_major_formatter(frmtr)
        # ax2.yaxis.set_major_formatter(frmtr)
        
        
        ax1.set_rlabel_position(22.5)  # Move grid labels away from other labels
        # ax2.set_rgrids([0.2, 0.4, 0.6, 0.8, 1])
        ax2.set_rlabel_position(22.5)  # Move grid labels away from other labels
        # plt.show()
        plt.legend(bbox_to_anchor=(0.3, 1.1))
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        #             hspace = 0, wspace = 0)
        # plt.margins(-0.1,-0.1)
        plt.savefig('responses.pdf', bbox_inches='tight')
        
    
    def get_beam_sources(self, input):
        features = self.get_features(input, squeeze_time=False) # angles, n_freqs, n_frames
        beam_sources = self.istft(features) # n_angles, n_samples
        return beam_sources
    
    def plot_beampatterns(self, freq=[1000, 3000]):
        theta_scan = np.linspace(0, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
        results1 = [[] for _ in range(self.K)]
        results2 = [[] for _ in range(self.K)]
        
        f1 = int(freq[0] / self.fs * self.n_fft)
        f2 = int(freq[1] / self.fs * self.n_fft)

        w1_Hs = [self.beamformer_weights_H[np.newaxis, beam_id, f1] for beam_id in range(self.K)] # each 1, C
        w2_Hs = [self.beamformer_weights_H[np.newaxis, beam_id, f2] for beam_id in range(self.K)] # each 1, C

        
        for theta_i in theta_scan:
            curr_s1 = np.array([np.exp(1j * (- 2 * np.pi * self.distances_acc[k] * np.cos(theta_i) / self.C * f1 / self.n_fft * self.fs)) for k in range(self.n_channels)]) # n_channels
            curr_s2 = np.array([np.exp(1j * (- 2 * np.pi * self.distances_acc[k] * np.cos(theta_i) / self.C * f2 / self.n_fft * self.fs)) for k in range(self.n_channels)]) # n_channels
            for beam_id in range(self.K):
                results1[beam_id].append((np.absolute((w1_Hs[beam_id] @ curr_s1)))[0])
                results2[beam_id].append((np.absolute((w2_Hs[beam_id] @ curr_s2)))[0])
        
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), subplot_kw={'projection': 'polar'})
        for beam_id in range(self.K):
            ax1.plot(theta_scan, results1[beam_id], alpha=0.7) # MAKE SURE TO USE RADIAN FOR POLAR
            ax2.plot(theta_scan, results2[beam_id], alpha=0.7)
            
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        ax2.set_thetamin(0)
        ax2.set_thetamax(180)
 
        ax1.set_title('1kHz')
        ax2.set_title('3kHz')
        ax1.set_theta_zero_location('W') # make 0 degrees point up
        ax1.set_theta_direction(-1) # increase clockwise
        ax2.set_theta_zero_location('W') # make 0 degrees point up
        ax2.set_theta_direction(-1) # increase clockwise
        # ax.set_rgrids([-10, -5, 0])
        # ax.set_rgrids([-8, -6, -4, -2, 0])
        # ax.set_rgrids([-40, -30, -20, -10, 0])
        ax1.set_rgrids([0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_rlabel_position(22.5)  # Move grid labels away from other labels
        ax2.set_rgrids([0.2, 0.4, 0.6, 0.8, 1])
        ax2.set_rlabel_position(22.5)  # Move grid labels away from other labels
        # plt.show()
        # plt.axis('off') 
        plt.savefig('beamform_180.pdf', bbox_inches='tight')

if __name__=='__main__':
    import os
    # import soundfile as sf
    beamformers = ISO_MVDR(k=10)
    beamformers.plot_beampatterns()
    # x = np.random.randn(8, 16000)
    # y = np.random.randn(8, 16000)

    # out = beamformers.cosine_similarity(x, x)

    # filename = '/data2/v_weiyangxu/inference_samples/reverb_clean_mc/sample_0.wav'
    # # filename = '/data2/lucayongxu/data/8mic_avlab_smallArray_aishell_dirDiffuseNoise_1spkAddSil_addEcho_fixNoiseGainIssue_multiChSpeech/outWavs/test/0-1.wav'
    # mc_audio, sr = librosa.load(filename, sr=16000, mono=False)
    # mc_audio = mc_audio[:8]
    # angle = np.load('gt_doa.npy')[1]
    # angle = 360 - angle #angle - 180

    # beamform_out = beamformers.beamform(mc_audio, angle)

    # sf.write('/data2/v_weiyangxu/inference_samples/test_beamform/sample_'+str(0)+'.wav', beamform_out, 16000)
    # beamformers.plot_beam_responses(mc_audio)
    
    samples_path = '/media/alan/新加卷/SpatialCodec/final_inference_samples'

    reverb_clean = 'reverb_clean_mc'

    reverb_clean_mc_opus_6kbps = 'reverb_clean_mc_opus_6kbps_merged'
    reverb_clean_mc_opus_12kbps = 'reverb_clean_mc_opus_12kbps_merged'
    reverb_clean_mc_opus_33kbps = 'reverb_clean_mc_opus_33kbps_merged'

    ref_hificodec_6kbps_reverb_clean = 'ref_hificodec_6kbps_reverb_clean'
    ref_encodec_6kbps_reverb_clean = 'ref_encodec_6kbps_reverb_clean'
    ref_subbandcodec_6kbps_reverb_clean = 'ref_subbandcodec_6kbps_reverb_clean'

    spatial_codec_est_clean = 'spatial_codec_est_clean'
    spatial_codec_est_clean_tuning = 'spatial_codec_est_clean_tuning'
    spatial_codec_cov_loss_est_clean_tuning = 'spatial_codec_cov_loss_est_clean_tuning'
    spatial_codec_MIMO = 'spatial_codec_MIMO'

    spatial_codec_est_clean_snr = 'spatial_codec_est_clean_snr'
    
    processed_types = [reverb_clean, reverb_clean_mc_opus_6kbps, reverb_clean_mc_opus_12kbps, ref_hificodec_6kbps_reverb_clean, ref_encodec_6kbps_reverb_clean, ref_subbandcodec_6kbps_reverb_clean, spatial_codec_est_clean, spatial_codec_est_clean_tuning, spatial_codec_cov_loss_est_clean_tuning, spatial_codec_MIMO, spatial_codec_est_clean_snr]
    processed_types = [reverb_clean, spatial_codec_MIMO, reverb_clean_mc_opus_12kbps, spatial_codec_est_clean]
    labels = ['groundtruth', 'E2E MIMO', 'opus_12', 'sub-band codec+SpatialCodec']
    processed_audios = []
    idx = 223
    beamformers = ISO_MVDR(k=50)
    # for i in range(200, 399):
    #     ref_audio, sr = librosa.load(os.path.join(samples_path, reverb_clean, 'sample_'+str(i)+'.wav'), sr=16000, mono=False)
    #     for processed_type in processed_types:
    #         mc_audio, sr = librosa.load(os.path.join(samples_path, processed_type, 'sample_'+str(i)+'.wav'), sr=16000, mono=False)
    #         score = beamformers.cosine_similarity(mc_audio, ref_audio)
    #         print(i, processed_type, score)
            
    for processed_type in processed_types:
        mc_audio, sr = librosa.load(os.path.join(samples_path, processed_type, 'sample_'+str(idx)+'.wav'), sr=16000, mono=False)
        processed_audios.append(mc_audio)
        
        
    beamformers.plot_beam_responses(processed_audios, labels)
    
    # beamformers.plot_beampatterns()
    
    
