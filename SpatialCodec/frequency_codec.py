import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.autograd import Variable

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

LRELU_SLOPE = 0.1

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 3), 1, dilation=(dilation[0], 1),
                               padding=(get_padding(kernel_size, dilation[0]), 1))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(dilation[1], 1),
                               padding=(get_padding(kernel_size, dilation[1]), 2))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(dilation[2], 1),
                               padding=(get_padding(kernel_size, dilation[2]), 2))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 3), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 1))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 2))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 2))),
        ])
        self.convs2.apply(init_weights)
        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
            
class Encoder(torch.nn.Module):
    def __init__(self, channels=[2, 16, 32, 64, 128, 128, 256]):
        super(Encoder, self).__init__()
        
        f_kernel_size = [5,3,3,3,3,4]
        f_stride_size = [2,2,2,2,2,1]
        resblock_kernel_sizes = [3,7]
        resblock_dilation_sizes = [[1,3,5], [1,3,5]]
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_layers = len(channels) - 1
        self.normalize = nn.ModuleList()

        conv_list = []
        norm_list = []
        res_list = []
        
        self.num_kernels = len(resblock_kernel_sizes)
        for c_idx in range(self.num_layers):
            conv_list.append(
                nn.Conv2d(channels[c_idx], channels[c_idx+1], (3, f_kernel_size[c_idx]), stride=(1, f_stride_size[c_idx]), padding=(1,0)),
            )
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(resblock_kernel_sizes)),
                    list(reversed(resblock_dilation_sizes))
                )
            ):
                res_list.append(ResBlock(channels[c_idx+1], k, d))    
                norm_list.append(nn.GroupNorm(1, channels[c_idx+1], eps=1e-6, affine=True))
       
        self.conv_list = nn.ModuleList(conv_list)
        self.norm_list = nn.ModuleList(norm_list)
        self.res_list = nn.ModuleList(res_list)
        
        self.conv_list.apply(init_weights)
        
        # self.lstm = nn.LSTM(512, 512, num_layers=2) # bs, T, 512
        
        # self.conv_post = weight_norm(nn.Conv1d(512, 512, 7, 1, padding=3))
        # self.conv_post.apply(init_weights)
        
        self.window = torch.hann_window(640)


    def forward(self, audio):
        '''
        x: bs, 2, T, F
        out: bs, 256, n_frames, 2
        '''
        audio = audio.squeeze(1)
        device = audio.device
        x = torch.stft(audio, n_fft=640, hop_length=320, win_length=640, window=self.window.to(device), center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        # x: bs, F, T
        x = torch.view_as_real(x).permute(0, 3, 2, 1)
        bs, _, n_frames, n_freqs = x.shape
        
        for i in range(self.num_layers):
            x = self.conv_list[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
                else:
                    xs += self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
            x = xs / self.num_kernels
            x = F.leaky_relu(x, LRELU_SLOPE) # bs, 256, n_frames, 2
        # bs, 256, n_frames, 9
        x = x.permute(0,3,1,2).reshape(bs, 6*256, n_frames)
        
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        
class Generator(torch.nn.Module):
    def __init__(self, channels=[16, 32, 64, 128, 256, 256, 256]):
        super(Generator, self).__init__()
        
        f_kernel_size = [5,3,3,3,3,4]
        f_stride_size = [2,2,2,2,2,1]
        resblock_kernel_sizes = [3, 7]
        resblock_dilation_sizes = [[1,3,5], [1,3,5]]
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_layers = len(channels) - 1
        self.normalize = nn.ModuleList()

        conv_list = []
        norm_list = []
        res_list = []
        
        self.num_kernels = len(resblock_kernel_sizes)
        for c_idx in range(self.num_layers):
            conv_list.append(
                nn.ConvTranspose2d(channels[self.num_layers-c_idx], channels[self.num_layers-c_idx-1], (3, f_kernel_size[self.num_layers-c_idx-1]), stride=(1, f_stride_size[self.num_layers-c_idx-1]), padding=(1,0)),
            )
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(resblock_kernel_sizes)),
                    list(reversed(resblock_dilation_sizes))
                )
            ):
                res_list.append(ResBlock(channels[self.num_layers-c_idx-1], k, d))    
                norm_list.append(nn.GroupNorm(1, channels[self.num_layers-c_idx-1], eps=1e-6, affine=True))
       
        self.conv_list = nn.ModuleList(conv_list)
        self.norm_list = nn.ModuleList(norm_list)
        self.res_list = nn.ModuleList(res_list)
        
        self.conv_list.apply(init_weights)
        self.conv_post = weight_norm(nn.Conv2d(16, 2, (5,5), (1,1), padding=(2,2)))
        self.conv_post.apply(init_weights)

        # self.conv_pre = weight_norm(nn.Conv1d(512, 512, 7, 1, padding=3))
        # self.conv_pre.apply(init_weights)
        
        # self.lstm = nn.LSTM(512, 512, num_layers=2) # bs, T, 512
        
        self.window = torch.hann_window(640)
        
    def forward(self, x):
        '''
        x: bs, 9*256, T
        out: bs, 
        '''
        bs, _, n_frames = x.shape
        # x = self.conv_pre(x)
        
        # x = x.permute(0,2,1)
        # x, _ = self.lstm(x) # bs, T, 512
        x = x.reshape(bs, 6, 256, n_frames)
        x = x.permute(0,2,3,1).contiguous()
        
        for i in range(self.num_layers):
            x = self.conv_list[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
                else:
                    xs += self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
            x = xs / self.num_kernels
            x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x) # bs, 2, T, F
        # bs, 2, T, F
        
        x = x.permute(0,3,2,1).contiguous()
        x = torch.istft(torch.view_as_complex(x), n_fft=640, hop_length=320, win_length=640, window=self.window.to(x.device), center=True, normalized=False, onesided=None, length=None, return_complex=False)
        
        x = x.unsqueeze(1)
        
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        
if __name__=='__main__':
    encoder = Encoder()
    decoder = Generator()
    input = torch.randn(3, 1, 16000*4)
    emb = encoder(input)
    print(emb.shape)
    out = decoder(emb)
    print(out.shape)
    
    x = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    y = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(x/1000000, y/1000000)
