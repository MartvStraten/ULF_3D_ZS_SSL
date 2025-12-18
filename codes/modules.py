import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import codes.utils as utils
import codes.mask_generator as mask_generator


class Dataset(torch.utils.data.Dataset):
    def __init__(self, trn_atb, trn_mask, loss_mask, ref_kspace):
        self.trn_atb = trn_atb
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.ref_kspace = ref_kspace
        
    def __len__(self):
        return len(self.trn_atb)
        
    def __getitem__(self, idx):
        nw_input = torch.tensor(self.trn_atb[idx], dtype=torch.float64) 
        nw_trn_mask = torch.tensor(self.trn_mask[idx], dtype=torch.complex64)
        nw_loss_mask = torch.tensor(self.loss_mask[idx], dtype=torch.complex64)
        ref_kspace = torch.tensor(self.ref_kspace[idx], dtype=torch.float64)

        return nw_input, nw_trn_mask, nw_loss_mask, ref_kspace

class Dataset_Inference(torch.utils.data.Dataset):
    def __init__(self, trn_atb, test_mask):
        self.trn_atb = trn_atb
        self.test_mask = test_mask
        
    def __len__(self):
        return len(self.trn_atb)
        
    def __getitem__(self, idx):
        nw_input = torch.tensor(self.trn_atb[idx], dtype=torch.float64)
        nw_test_mask = torch.tensor(self.test_mask[idx], dtype=torch.complex64)

        return nw_input, nw_test_mask

class Augment3D:
    def __init__(self, 
        rotation_range=10,
        translation_range=5,
        flip_prob=0.25,
        scale_range=0.1
        ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.flip_prob = flip_prob
        self.scale_range = scale_range

    def random_flip(self, x_re, x_im):
        # Horizontal flip
        if random.random() < self.flip_prob:
            x_re = torch.flip(x_re, dims=[-1])
            x_im = torch.flip(x_im, dims=[-1])  
        # Vertical flip
        if random.random() < self.flip_prob:
            x_re = torch.flip(x_re, dims=[-2])
            x_im = torch.flip(x_im, dims=[-2]) 
        # Depth flip
        if random.random() < self.flip_prob:
            x_re = torch.flip(x_re, dims=[-3])
            x_im = torch.flip(x_im, dims=[-3])

        return x_re, x_im

    def random_affine(self, x_re, x_im):
        """Apply random 3D rotation using affine grid."""
        # Determine rotation angle and translation pixels
        angle_x = np.deg2rad(random.uniform(-self.rotation_range, self.rotation_range))
        angle_y = np.deg2rad(random.uniform(-self.rotation_range, self.rotation_range))
        angle_z = np.deg2rad(random.uniform(-self.rotation_range, self.rotation_range))
        tx = random.uniform(-self.translation_range, self.translation_range)
        ty = random.uniform(-self.translation_range, self.translation_range)
        tz = random.uniform(-self.translation_range, self.translation_range)

        # Define affine transformation matrix
        Rx = torch.tensor([[1, 0, 0],
                           [0, np.cos(angle_x), -np.sin(angle_x)],
                           [0, np.sin(angle_x),  np.cos(angle_x)]])
        Ry = torch.tensor([[np.cos(angle_y), 0, np.sin(angle_y)],
                           [0, 1, 0],
                           [-np.sin(angle_y), 0, np.cos(angle_y)]])
        Rz = torch.tensor([[np.cos(angle_z), -np.sin(angle_z), 0],
                           [np.sin(angle_z),  np.cos(angle_z), 0],
                           [0, 0, 1]])
        R = Rz @ Ry @ Rx
        R_full = torch.cat(
            [R, torch.tensor([
                [tx / (x_re.shape[-1]/2)], 
                [ty / (x_re.shape[-2]/2)], 
                [tz / (x_re.shape[-3]/2)]])], 
            dim=1)
        R_full = R_full[None].to(x_re.device, dtype=x_re.dtype)
        
        # Apply affine transformation to real and imag components
        grid = F.affine_grid(R_full, size=(1, 1, *x_re.shape[-3:]), align_corners=False)
        x_re_rot = F.grid_sample(x_re[None,None], grid, mode='bilinear', align_corners=False)
        x_im_rot = F.grid_sample(x_im[None,None], grid, mode='bilinear', align_corners=False)

        return x_re_rot[0,0], x_im_rot[0,0]

    def random_scale(self, x_re, x_im):
        # Determine scaling factor
        scale_factor = 1 + random.uniform(-self.scale_range, self.scale_range)
        new_size = [int(dim * scale_factor) for dim in x_re.shape[-3:]]
        
        # Apply scaling to real
        x_re_scaled = F.interpolate(x_re[None,None], size=new_size, mode='trilinear', align_corners=False)
        x_re_scaled = F.interpolate(x_re_scaled, size=x_re.shape[-3:], mode='trilinear', align_corners=False)
        
        x_im_scaled = F.interpolate(x_im[None,None], size=new_size, mode='trilinear', align_corners=False)
        x_im_scaled = F.interpolate(x_im_scaled, size=x_im.shape[-3:], mode='trilinear', align_corners=False)
        
        return x_re_scaled[0,0], x_im_scaled[0,0]

    def __call__(self, x_re, x_im):
        x_re, x_im = self.random_flip(x_re, x_im)
        x_re, x_im = self.random_affine(x_re, x_im)
        x_re, x_im = self.random_scale(x_re, x_im)
        
        return x_re, x_im

class Dataset_Pretrain(torch.utils.data.Dataset):
    def __init__(self, ref_kspace, args, augment=True):
        self.ref_kspace = ref_kspace
        self.args = args
        self.augment = augment
        self.augmenter = Augment3D() if augment else None

        # Mask generation parameters
        self.accel_rate = args.acc_rate
        self.cRP = 0.125
        
    def __len__(self):
        return self.ref_kspace.shape[0]
        
    def __getitem__(self, idx):
        # Undersampled image (input)
        ref_kspace = self.ref_kspace[idx]

        # Data augmentation
        if self.augment:
            ref_image = utils.ifftc2(ref_kspace, axes=(-1,-2))
            ref_image_real = torch.tensor(utils.c2r(ref_image, axis=0), dtype=torch.float64)
            aug_image_re, aug_image_im = self.augmenter(ref_image_real[0], ref_image_real[1])
            aug_image_real = torch.stack((aug_image_re, aug_image_im))
            aug_image = utils.r2c(aug_image_real, axis=0)
            ref_kspace = utils.fftc2(aug_image, axes=(-1,-2)).numpy()
        
        # Generate mask
        mask, true_accel_rate = mask_generator.generate_pdf_mask(
            self.args.ndepth, self.args.ncol, accel=self.accel_rate, radius=self.cRP
        )

        # Prepare tensors
        nw_input = utils.ifftc2(ref_kspace * mask[None], axes=(-1,-2))
        ref_kspace_real = utils.c2r(ref_kspace, axis=0)
        nw_input_real = utils.c2r(nw_input, axis=0)

        # Prepare tensors
        nw_input_real = torch.tensor(nw_input_real, dtype=torch.float64)
        nw_mask = torch.tensor(mask[None], dtype=torch.complex64)
        nw_loss_mask = torch.ones(nw_mask.shape, dtype=torch.complex64)
        ref_kspace_real = torch.tensor(ref_kspace_real, dtype=torch.float64)

        return nw_input_real, nw_mask, nw_loss_mask, ref_kspace_real
    
class Dataset_Inference_Pretrain(torch.utils.data.Dataset):
    def __init__(self, ref_kspace, args):
        self.ref_kspace = ref_kspace
        self.args = args

        # Mask generation parameters
        self.accel_rate = args.acc_rate
        self.cRP = 0.125
        
    def __len__(self):
        return self.ref_kspace.shape[0]
        
    def __getitem__(self, idx):
        # Generate mask
        mask, true_accel_rate = mask_generator.generate_pdf_mask(
            self.args.ndepth, self.args.ncol, accel=self.accel_rate, radius=self.cRP
        )

        # Undersampled image (input)
        ref_kspace = self.ref_kspace[idx]
        nw_input = utils.ifftc2(ref_kspace*mask[None], axes=(-1,-2))
        
        # Prepare real-valued tensors
        nw_input_real = utils.c2r(nw_input, axis=0)

        # Prepare tensors
        nw_input_real = torch.tensor(nw_input_real, dtype=torch.float64)
        nw_mask = torch.tensor(mask[None], dtype=torch.complex64)

        return nw_input_real, nw_mask


class MixL1L2Loss(nn.Module):
    def __init__(self, eps=1e-6, scalar=0.5):
        super().__init__()
        self.eps = eps
        self.scalar = scalar

    def forward(self, yhat, y):
        loss = self.scalar * (torch.norm(yhat - y) / torch.norm(y)) + self.scalar * (torch.norm(yhat - y, p=1) / torch.norm(y, p=1))

        return loss

def train(train_loader, model, loss_fn, optimizer, device):
    avg_trn_cost = 0
    scaler = torch.amp.GradScaler("cuda")
    model.train()

    for ii, batch in enumerate(train_loader):
        nw_input, nw_trn_mask, nw_loss_mask, nw_ref_kspace = batch
        nw_input, nw_trn_mask, nw_loss_mask, nw_ref_kspace = \
            nw_input.to(device), nw_trn_mask.to(device), nw_loss_mask.to(device), nw_ref_kspace.to(device)

        optimizer.zero_grad()

        # Forward path
        with torch.amp.autocast(device_type="cuda"):
            nw_img_output, lamdas, nw_kspace_output = model(nw_input, nw_trn_mask, nw_loss_mask)
            trn_loss = loss_fn(nw_kspace_output, nw_ref_kspace)
        
        # Backpropagationnw_kspace_output
        scaler.scale(trn_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        avg_trn_cost += trn_loss.item()/ len(train_loader)

    return avg_trn_cost, lamdas, nw_kspace_output, nw_ref_kspace

def validation(val_loader, model, loss_fn, device=torch.device('cpu')):
    avg_val_cost = 0
    model.eval()
    with torch.no_grad():
        for ii,batch in enumerate(val_loader):
            nw_input, nw_trn_mask, nw_loss_mask, nw_ref_kspace = batch
            nw_input, nw_trn_mask, nw_loss_mask, nw_ref_kspace = \
                nw_input.to(device), nw_trn_mask.to(device), nw_loss_mask.to(device), nw_ref_kspace.to(device)

            # Forward path
            with torch.amp.autocast(device_type="cuda"):
                nw_img_output, lamdas, nw_kspace_output = model(nw_input, nw_trn_mask, nw_loss_mask)
                val_loss = loss_fn(nw_kspace_output, nw_ref_kspace)
            
            avg_val_cost += val_loss.item() / len(val_loader)

    return avg_val_cost, nw_kspace_output, nw_ref_kspace

def test(test_loader, model, device=torch.device('cpu')):
    model.eval()

    with torch.no_grad():
        for ii,batch in enumerate(test_loader):
            nw_input, nw_test_mask = batch 
            nw_input, nw_test_mask = nw_input.to(device), nw_test_mask.to(device)

            # Forward path
            with torch.amp.autocast(device_type="cuda"):
                nw_img_output, lamdas, nw_kspace_output = model(nw_input, nw_test_mask, nw_test_mask)

    return nw_img_output
