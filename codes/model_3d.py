import torch.nn as nn
import torch
import math

from codes.utils import ifftc2, fftc2, r2c, c2r


# Residual block =====

def activation_func(activation):
    return nn.ModuleDict([
        ['ReLU', nn.ReLU()],
        ['LeakyReLU', nn.LeakyReLU(0.1)],
        ['None', nn.Identity()]
    ])[activation]

def conv_layer(filter_size, padding=1, batchnorm=False, activation_type='LeakyReLU'):
    kernel_size, in_channels, out_channels = filter_size
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False), 
        nn.BatchNorm3d(out_channels) if batchnorm else nn.Identity(),
        activation_func(activation_type)
    )

def ResNetBlock(filter_size):
    return nn.Sequential(
        conv_layer(filter_size, activation_type='LeakyReLU'),
        conv_layer(filter_size, activation_type='None')
    )
    
class ResNetBlocksModule(nn.Module):
    def __init__(self, device, filter_size, num_blocks, residual_scale=0.1, time_emb_dim=128):
        super().__init__()
        self.device = device
        self.time_emb_dim = time_emb_dim

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim, max_period=num_blocks)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, filter_size[-1]),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([ResNetBlock(filter_size=filter_size) for _ in range(num_blocks)])
        self.register_buffer('scale', torch.tensor([residual_scale], dtype=torch.float32))

    def forward(self, x, t):
        # Project and reshape timestep to (B, C, 1, 1, 1)
        time_emb = self.time_embed(t)
        timestep = self.time_proj(time_emb)[..., None, None, None]
        x = x + timestep
        
        # Perform ResNet block
        for layer in self.layers:
            x = x + layer(x) * self.scale
            
        return x
    
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_period):
        """Creates a sinusoidal embedding for time steps.
        ---
        Parameters
          embedding_dim (int): dimension of initial sinusoidal embedding.
          max_period (int): maximum period in sinusoidal embedding.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.half_dim = self.embedding_dim // 2

    def forward(self, timesteps):
        """Returns the sinusoidal embedding for a given time step."""
        self.freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=self.half_dim, dtype=torch.float) / (self.half_dim - 1)
        ).to(timesteps.device)

        # Create sinusoidal time-step embeddings
        args = timesteps[:, None].float() * self.freqs[None] # Shape (batch, half_dim)
        embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1) # Shape (batch, embedding_dim)

        return embedding

class ResNet(nn.Module):
    def __init__(self, device, in_channels, num_resblocks):
        super(ResNet, self).__init__()
        kernel_size = 3
        features = 64

        filter1 = [kernel_size, in_channels, features] # Map input to size of feature maps
        filter2 = [kernel_size, features, features] # ResNet Blocks
        filter3 = [kernel_size, features, in_channels] # Map output channels to input channels

        self.layer1 = conv_layer(filter_size=filter1, activation_type='None')
        self.layer2 = ResNetBlocksModule(device=device, filter_size=filter2, num_blocks=num_resblocks)
        self.layer3 = conv_layer(filter_size=filter3, activation_type='None')

    def forward(self, input_x, t):
        l1 = self.layer1(input_x)
        l2 = self.layer2(l1, t)
        nw_out = self.layer3(l1 + l2)
        return nw_out
    

# Data consistency =====

class data_consistency():
    """Class used to perform data consistency block."""
    def __init__(self, mask, args):
        self.mask = mask
        self.args = args

    def EhE_Op(self, img, mu):
        """Performs (E^h*E + mu*I)x """
        kspace = fftc2(img, axes=(-1,-2))
        masked_kspace = kspace * self.mask
        masked_ispace = ifftc2(masked_kspace, axes=(-1,-2)) 
        ispace = masked_ispace + mu*img

        return ispace

    def SSDU_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        """
        kspace = fftc2(img, axes=(-1,-2))
        masked_kspace = kspace * self.mask

        return masked_kspace
    
def zdot_reduce_sum(input_x, input_y):
    dims = tuple(range(len(input_x.shape)))

    return (torch.conj(input_x)*input_y).sum(dims).real

def conjgrad(rhs, mask, mu, args): 
    """Performs conjugate gradient algorithm on data."""
    Encoder = data_consistency(mask, args)
    rhs = r2c(rhs.reshape(args.ncoil*args.ncontrast, 2, args.nrow, args.ncol, args.ndepth), axis=1)
    mu = mu.type(torch.complex64)

    x = torch.zeros_like(rhs)
    r, p = rhs, rhs
    rsnot = zdot_reduce_sum(r, r)
    rsold, rsnew = rsnot, rsnot

    for _ in range(args.CG_Iter):
        Ap = Encoder.EhE_Op(p, mu)
        pAp = zdot_reduce_sum(p, Ap)
        alpha = (rsold / pAp)
        x = x + alpha*p
        r = r - alpha*Ap
        rsnew = zdot_reduce_sum(r, r)
        beta = (rsnew / rsold)
        rsold = rsnew
        p = beta * p + r

    return torch.reshape(c2r(x, axis=1), (2*args.ncoil*args.ncontrast, args.nrow, args.ncol, args.ndepth))

def dc_block(rhs, mask, mu, args):
    """DC block employs conjugate gradient for data consistency."""
    cg_recons = []
    for i in range(args.batchSize):
        cg_recon = conjgrad(rhs[i], mask[i], mu, args)
        cg_recons.append(cg_recon.unsqueeze(0))

    return torch.cat(cg_recons, 0)

def SSDU_kspace_transform(nw_output, mask, args):
    """Transforms unrolled network output to k-space at only unseen locations in training."""
    all_recons = []
    for i in range(args.batchSize):
        Encoder = data_consistency(mask[i], args)
        temp = r2c(nw_output[i].reshape(args.ncoil*args.ncontrast, 2, args.nrow, args.ncol, args.ndepth), axis=1)
        nw_output_kspace = Encoder.SSDU_kspace(temp) 
        nw_output_kspace = torch.reshape(c2r(nw_output_kspace, axis=1), (2*args.ncoil*args.ncontrast, args.nrow, args.ncol, args.ndepth))
        all_recons.append(nw_output_kspace.unsqueeze(0))

    return torch.cat(all_recons, 0)


# Unrolled network =====

class UnrolledNet(nn.Module):
    """Class used to define the full unrolled network."""
    def __init__(self, args, device):
        super(UnrolledNet, self).__init__()
        self.args = args
        self.device = device
        self.lamda_start = 0.05
        in_ch = 2*args.ncoil*args.ncontrast

        self.regularizer = ResNet(device, in_channels=in_ch, num_resblocks=args.nb_res_blocks) 
        self.lam = torch.tensor([self.lamda_start], device=device)
    
    def forward(self, input_x, trn_mask, loss_mask):
        """Performs unrolled optimization process."""
        x = input_x # Shape (nb, nc, nx, ny, nz)
        nbatch = x.shape[0]
        
        for n in range(self.args.nb_unroll_blocks):
            # Prepare time-step tensor
            t = torch.tensor([n] * nbatch, device=x.device)

            # Apply regularizer
            x = self.regularizer(x.float(), t)

            # Data consistency
            rhs = input_x + self.lam * x
            x = dc_block(rhs, trn_mask, self.lam, self.args)

        nw_kspace_output = SSDU_kspace_transform(x, loss_mask, self.args)

        return x, self.lam, nw_kspace_output