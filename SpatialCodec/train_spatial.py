# from hificodec train.py @ https://github.com/yangdongchao/AcademiCodec
# pipeline for training rnnbf-hificodec
# weiyang 2023-05-26

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
# from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from mc_mel_dataset import MelDataset, get_audio_files, mel_spectrogram
from msstftd import MultiScaleSTFTDiscriminator
from models import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss, Quantizer
try:
    from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
except:
    from .utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
    
from rnnbf_model.RNNBF import RNNBF
from spatial_model_subband import SpatialEncoder, SpatialDecoder

# from loss import loss_l1_wav_mag, rtf_emb_loss, snr, rtf_l2_loss

def snr(y, y_hat, epsilon=1e-8):
    '''
    s: bs, T
    s_hat: bs, T
    '''
    return 10*torch.log10((y**2).sum(1)+epsilon) - 10*torch.log10(((y - y_hat)**2).sum(1)+epsilon)

torch.backends.cudnn.benchmark = True

def reconstruction_loss(x, G_x, device, eps=1e-7):
    L = 100*F.mse_loss(x, G_x) # wav L1 loss
    for i in range(6,11):
        s = 2**i
        melspec = MelSpectrogram(sample_rate=24000, n_fft=s, hop_length=s//4, n_mels=64, wkwargs={"device": device}).to(device)
        # 64, 16, 64
        # 128, 32, 128
        # 256, 64, 256
        # 512, 128, 512
        # 1024, 256, 1024
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        loss = ((S_x-S_G_x).abs().mean() + (((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2).mean(dim=-2)**0.5).mean())/(i)
        L += loss
        #print('i ,loss ', i, loss)
    #assert 1==2
    return L

def train(rank, a, h):
    if h.num_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12398'
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    # encoder = Encoder(h).to(device)
    # generator = Generator(h).to(device)
    # quantizer = Quantizer(h).to(device)
    quantizer = Quantizer(h, n_dim=256*6).to(device)
    Encoder = SpatialEncoder().to(device)
    Decoder = SpatialDecoder().to(device)

    ckpt = torch.load('/apdcephfs_cq2/share_1603164/data/weiyangxu/from_36/spatial_codec_cov_suband_coding_full_conv_6kbps_deepfilter/logs/g_00270000', map_location=device)
    Encoder.load_state_dict(ckpt['encoder'], strict=False)
    Decoder.load_state_dict(ckpt['decoder'], strict=False)
    quantizer.load_state_dict(ckpt['quantizer'], strict=False)

    if rank == 0:
        # print(rnnbf)
        print(Encoder)
        print(Decoder)
        print(quantizer)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    steps = 0
    last_epoch = -1

    if h.num_gpus > 1:
        Encoder = DistributedDataParallel(Encoder, device_ids=[rank]).to(device)
        Decoder = DistributedDataParallel(Decoder, device_ids=[rank]).to(device)
        quantizer = DistributedDataParallel(quantizer, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(itertools.chain(
        Encoder.parameters(), Decoder.parameters(), quantizer.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
        
    # if state_dict_do is not None:
    #     optim_g.load_state_dict(state_dict_do['optim_g'])
    #     optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # training_filelist, validation_filelist = get_dataset_filelist(a)
    training_audiofiles = get_audio_files(
        inpath=a.in_path,
        scp_path=a.scp_path,
        mode='train')
    validation_audiofiles = get_audio_files(
        inpath=a.in_path,
        scp_path=a.scp_path,
        mode='test')
    
    trainset = MelDataset(
        audio_files = training_audiofiles,
        segment_dur = h.segment_dur,
        mono=h.mono,
        n_fft=h.n_fft, num_mels=h.num_mels,
        hop_size=h.hop_size,
        win_size=h.win_size,
        sampling_rate=h.sampling_rate,
        fmin=h.fmin,
        fmax=h.fmax,
        shuffle=False if h.num_gpus > 1 else True,
        device=device, fmax_loss=h.fmax_for_loss, fine_tuning=a.fine_tuning,
        random_start=True
        )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(
                audio_files = validation_audiofiles,
                segment_dur = h.segment_dur,
                mono=h.mono,
                n_fft=h.n_fft, num_mels=h.num_mels,
                hop_size=h.hop_size,
                win_size=h.win_size,
                sampling_rate=h.sampling_rate,
                fmin=h.fmin,
                fmax=h.fmax,
                shuffle=False if h.num_gpus > 1 else True,
                device=device, fmax_loss=h.fmax_for_loss, fine_tuning=a.fine_tuning,
                random_start=True
            )
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                        sampler=None,
                                        batch_size=1,
                                        pin_memory=True,
                                        drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    Encoder.train()
    Decoder.train()
    quantizer.train()
    # mpd.train()
    # msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):

            if rank == 0:
                start_b = time.time()
            # x, y, _, y_mel = batch # mel, audio, _, mel_loss
            # noisy_array, x, y, _, y_mel = batch # noisy_multichannel, mel, audio, _, mel_loss
            _, reverb_clean, _, cov, ref_stft, vad_frames = batch # 
                        
            reverb_clean = torch.autograd.Variable(reverb_clean.to(device, non_blocking=True)) # bs, 8, n_samples
            cov = torch.autograd.Variable(cov.to(device, non_blocking=True)) # bs, 72, F, T
            vad_frames = torch.autograd.Variable(vad_frames.to(device, non_blocking=True)) # bs, n_frames
            ref_stft = torch.autograd.Variable(ref_stft.to(device, non_blocking=True)) # [bs, 1, 321, n_frames, 2]
            
            bs, _, n_freqs, n_frames = cov.shape # bs, 72, F, T
        
            # encoder input: # bs, 72, F, T
            cov = cov.permute(0,1,3,2)
            c_emb = Encoder(cov) # bs, 9*512, n_frames

            q_rtf, loss_q_rtf, _ = quantizer(c_emb) # bs, 22*512, n_frames

            # decode
            rtf_g = Decoder(q_rtf) # bs, 7, n_freqs, 5, n_frames, 2
            
            ref_stft = torch.view_as_complex(ref_stft) # bs, 7, F, n_frames
            # print(ref_stft.shape, F.unfold(ref_stft, kernel_size=(n_freqs, 5), padding=(0,2)).shape)

            ref_stft_df = F.unfold(ref_stft, kernel_size=(3, 9), padding=(1,4)).reshape(bs, 1, 3,9,n_freqs, n_frames) # bs, 7, 3, 9, n_freqs, n_frames
            rtf_g = torch.view_as_complex(rtf_g).contiguous() # [2, 7, 3, 9, 321, 100]
            est_stfts = (rtf_g * ref_stft_df).sum(2).sum(2) # bs, 7, n_freqs, n_framesest_stfts
            # est_stfts = rtf_g * ref_stft
            est_stfts = est_stfts.reshape(bs*7, n_freqs, n_frames)
            est_reverb_clean = torch.istft(est_stfts, 640, 320, 640, window=trainset.hann_window.to(est_stfts.device)) # bs*7, T
            
            reverb_clean = reverb_clean[:, [0,1,2,4,5,6,7], :].reshape(bs*7, -1)
            # Generator
            optim_g.zero_grad()
            
            # loss_wav_reconstruct = 300*loss_l1_wav_mag(reverb_clean, est_reverb_clean)
            loss_snr = - snr(reverb_clean, est_reverb_clean).mean()
            
            loss_gen_all = (loss_q_rtf) * 10 + loss_snr
            # print(loss_rtf, loss_q, loss_wav_reconstruct)
            
            loss_gen_all.backward()
            optim_g.step()
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    # print('Steps : {:d}, loss_rtf : {:4.3f}, loss_rtf_emb : {:4.3f}, Loss Q : {:4.3f}, loss_wav_reconstruct : {:4.3f}, loss_gen_all : {:4.3f}, snr : {:4.3f}, s/b : {:4.3f}'.
                    #       format(steps, loss_rtf, loss_rtf_emb, loss_q, loss_wav_reconstruct, loss_gen_all, -loss_snr, time.time() - start_b))
                    print('Steps : {:d}, loss_gen_all : {:4.3f}, snr : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, -loss_snr, time.time() - start_b))
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                # if True:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'decoder': (Decoder.module if h.num_gpus > 1 else Decoder).state_dict(),
                                     'encoder': (Encoder.module if h.num_gpus > 1 else Encoder).state_dict(),
                                     'quantizer': (quantizer.module if h.num_gpus > 1 else quantizer).state_dict(),
                                     }, num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
    
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    # sw.add_scalar("training/loss_rtf", loss_rtf, steps)
                    # sw.add_scalar("training/loss_rtf_emb", loss_rtf_emb, steps)
                    # sw.add_scalar("training/loss_q", loss_q, steps)
                    # sw.add_scalar("training/loss_reconstruct", loss_wav_reconstruct, steps)
                    sw.add_scalar("training/snr", -loss_snr, steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                # if True:
                    Decoder.eval()
                    Encoder.eval()
                    quantizer.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            # print(j)
                            # if j > 1:
                            #     break
                            # -------------------------------------
                            _, reverb_clean, _, cov, ref_stft, vad_frames = batch # 
                        
                            reverb_clean = torch.autograd.Variable(reverb_clean.to(device, non_blocking=True)) # bs, 8, n_samples
                            cov = torch.autograd.Variable(cov.to(device, non_blocking=True)) # bs, 7, 321, n_frames, 2
                            vad_frames = torch.autograd.Variable(vad_frames.to(device, non_blocking=True)) # bs, n_frames
                            ref_stft = torch.autograd.Variable(ref_stft.to(device, non_blocking=True)) # [bs, 1, 321, n_frames, 2]
                            
                            bs, _, n_freqs, n_frames = cov.shape # bs, 72, 321, n_frames, 2
                        
                            # encoder input: bs, 2, 7, F         
                            cov = cov.permute(0,1,3,2)  
                            c_emb = Encoder(cov) # bs, 22*512, n_frames

                            q_rtf, loss_q_rtf, _ = quantizer(c_emb) # bs, 22*512, n_frames

                            # decode
                            rtf_g = Decoder(q_rtf) # bs, 7, n_freqs, 5, n_frames, 2
                            
                            ref_stft = torch.view_as_complex(ref_stft) # bs, 7, F, n_frames
                            # print(ref_stft.shape, F.unfold(ref_stft, kernel_size=(n_freqs, 5), padding=(0,2)).shape)

                            ref_stft_df = F.unfold(ref_stft, kernel_size=(3, 9), padding=(1,4)).reshape(bs, 1, 3,9,n_freqs, n_frames) # bs, 7, 3, 9, n_freqs, n_frames
                            rtf_g = torch.view_as_complex(rtf_g).contiguous() # [2, 7, 3, 9, 321, 100]
                            est_stfts = (rtf_g * ref_stft_df).sum(2).sum(2) # bs, 7, n_freqs, n_framesest_stfts
                            est_stfts = est_stfts.reshape(bs*7, n_freqs, n_frames)
                            est_reverb_clean = torch.istft(est_stfts, 640, 320, 640, window=trainset.hann_window.to(est_stfts.device)) # bs*7, T
                            
                            reverb_clean = reverb_clean[:, [0,1,2,4,5,6,7], :].reshape(bs*7, -1)
                            # -----------------------------    
                            snr_vals = snr(reverb_clean, est_reverb_clean).mean()
                            print('val snr: ', snr_vals)
                            val_err_tot += snr_vals.item()

                            if j <= 8:
                                # if steps == 0:
                                sw.add_audio('gt/y_{}'.format(j), reverb_clean[0], steps, h.sampling_rate)
                                y_spec = mel_spectrogram(reverb_clean[0].unsqueeze(0), h.n_fft, h.num_mels,
                                                            h.sampling_rate, h.hop_size, h.win_size,
                                                            h.fmin, h.fmax)
                                sw.add_figure('gt/y_spec_{}'.format(j), 
                                            plot_spectrogram(y_spec.squeeze(0).cpu().numpy()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), est_reverb_clean[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(est_reverb_clean[0].unsqueeze(0), h.n_fft, h.num_mels,
                                                            h.sampling_rate, h.hop_size, h.win_size,
                                                            h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                            plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/snr", val_err, steps)

                        Encoder.train()
                        Decoder.train()
                        quantizer.train()
                    
            steps += 1

        scheduler_g.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    # parser.add_argument('--group_name', default=None)
    # parser.add_argument('--input_wavs_dir', default='../datasets/audios')
    parser.add_argument('--input_mels_dir', default=None)
    parser.add_argument('--in_path', required=True)
    parser.add_argument('--scp_path', required=True)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=5, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
