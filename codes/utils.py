import numpy as np
import torch
import random
import math

from skimage.metrics import structural_similarity


# Utilities =====

def set_seeds(seed):
    """Sets seed for all random processes."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):
    nc, nx, ny = input_data.shape

    center_kx = int(find_center_ind(input_data, axes=(0, 2)))
    center_ky = int(find_center_ind(input_data, axes=(0, 1)))

    temp_mask = np.copy(input_mask)
    temp_mask[:, center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
        center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(
        np.arange(nc * nx * ny),
        size=int(np.count_nonzero(pr) * rho), 
        replace=False, 
        p=pr / np.sum(pr)
    )

    [ind_c, ind_x, ind_y] = index_flatten2nd(ind, (nc, nx, ny))

    loss_mask = np.zeros_like(input_mask)
    loss_mask[ind_c, ind_x, ind_y] = 1

    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask


def uniform_selection_3d(input_data, input_mask, rho=0.2, small_acs_block=(6, 6)):
    nc, nx, ny, nz = input_data.shape

    center_ky = int(find_center_ind(input_data, axes=(0, 1, 3)))
    center_kz = int(find_center_ind(input_data, axes=(0, 1, 2)))

    temp_mask = np.copy(input_mask)
    temp_mask[..., center_ky - small_acs_block[0] // 2: center_ky + small_acs_block[0] // 2,
        center_kz - small_acs_block[1] // 2: center_kz + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(
        np.arange(nc * ny * nz),
        size=int(np.count_nonzero(pr) * rho), 
        replace=False, 
        p=pr / np.sum(pr)
    )

    [ind_c, ind_y, ind_z] = index_flatten2nd(ind, (nc, ny, nz))

    loss_mask = np.zeros_like(input_mask)
    loss_mask[ind_c, 0, ind_y, ind_z] = 1

    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask


def find_center_ind(kspace, axes=(1, 2, 3)):
    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: 
        return tensor.squeeze()

    return tensor


def index_flatten2nd(ind, shape):
    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


# Maths =====

def fftc1(image, axis=(-1), norm="ortho"):
    """Calculates 1D FFT.
    ---
    Parameters
      image: data to be 1D-Fourier transformed.
      axis (tuple): define the axes which are transformed.
    --- 
    Returns
      kspace: data after 1D-Fourier transformation.
    """
    if isinstance(image, np.ndarray):
        kspace = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(image, axes=axis), axis=axis, norm=norm), axes=axis)
    elif isinstance(image, torch.Tensor):
        kspace = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(image, dim=axis), dim=axis, norm=norm), dim=axis)
    else:
        raise NotImplementedError

    return kspace

def ifftc1(kspace, axis=(-1), norm="ortho"):
    """Calculates inverse 1D FFT.
    ---
    Parameters
      kspace: data to be inversely 1D-Fourier transformed.
      axis (tuple): define the axes which are transformed. 
    ---  
    Returns
      image: data after inverse 1D-Fourier transformation.
    """
    if isinstance(kspace, np.ndarray):
        image = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace, axes=axis), axis=axis, norm=norm), axes=axis)
    elif isinstance(kspace, torch.Tensor):
        image = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(kspace, dim=axis), dim=axis, norm=norm), dim=axis)
    else:
        raise NotImplementedError

    return image

def fftc2(image, axes=(-2,-1), norm="ortho"):
    """Calculates 2D FFT.
    ---
    Parameters
      image: data to be 2D-Fourier transformed.
      axis (tuple): define the axes which are transformed.
    --- 
    Returns
      kspace: data after 2D-Fourier transformation.
    """
    if isinstance(image, np.ndarray):
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes, norm=norm), axes=axes)
    elif isinstance(image, torch.Tensor):
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image, dim=axes), dim=axes, norm=norm), dim=axes)
    else:
        raise NotImplementedError

    return kspace


def ifftc2(kspace, axes=(-2,-1), norm="ortho"):
    """Calculates inverse 2D FFT.
    ---
    Parameters
      kspace: data to be inversely 2D-Fourier transformed.
      axis (tuple): define the axes which are transformed. 
    ---  
    Returns
      image: data after inverse 2D-Fourier transformation.
    """
    if isinstance(kspace, np.ndarray):
        image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)
    elif isinstance(kspace, torch.Tensor):
        image = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=axes), dim=axes, norm=norm), dim=axes)
    else:
        raise NotImplementedError

    return image


def fftcn(image, axes=(-3,-2,-1)):
    """Calculates 3D FFT.
    ---
    Parameters
      image: data to be 3D-Fourier transformed.
      axis (tuple): define the axes which are transformed.
    --- 
    Returns
      kspace: data after 3D-Fourier transformation.
    """
    if isinstance(image, np.ndarray):
        kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes)
    elif isinstance(image, torch.Tensor):
        kspace = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image, dim=axes), dim=axes), dim=axes)
    else:
        raise NotImplementedError
    
    return kspace


def ifftcn(kspace, axes=(-3,-2,-1)):
    """Calculates inverse 3D FFT.
    ---
    Parameters
      kspace: data to be inversely 3D-Fourier transformed.
      axis (tuple): define the axes which are transformed. 
    ---  
    Returns
      image: data after inverse 3D-Fourier transformation.
    """
    if isinstance(kspace, np.ndarray):
        image = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes), axes=axes)
    elif isinstance(kspace, torch.Tensor):
        image = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=axes), dim=axes), dim=axes)
    else:
        raise NotImplementedError

    return image


def c2r(complex_data, axis=-1):
    """Convert complex-valued data to real-valued data.
    ---
    Parameters:
      complex_data: a complex-valued np.ndarray or torch.tensor (shape).
      axis (int): axis with real and imaginary parts.
    ---
    Returns:
      real_data: a real-valued np.ndarray or torch.tensor (shape, 2).
    """
    if isinstance(complex_data, np.ndarray):
        real_data = np.stack((complex_data.real, complex_data.imag), axis=axis)
    elif isinstance(complex_data, torch.Tensor):
        real_data = torch.stack((complex_data.real, complex_data.imag), dim=axis).to(torch.float)
    else:
        raise NotImplementedError("Input must be a NumPy array or PyTorch tensor.")
    
    return real_data


def r2c(real_data, axis=-1):
    """Convert real-valued data to complex-valued data.
    ---
    Parameters:
      real_data: a real-valued np.ndarray or torch.tensor (shape, 2).
      axis (int): axis with real and imaginary parts.
    ---
    Returns:
      complex_data: a complex-valued np.ndarray or torch.tensor (shape).
    """    
    if isinstance(real_data, np.ndarray):
        if real_data.shape[axis] != 2:
            raise ValueError(f"Expected size-2 along axis {axis}, got shape {real_data.shape}")
        
        real = np.take(real_data, 0, axis=axis)
        imag = np.take(real_data, 1, axis=axis)
        return real + 1j * imag

    elif isinstance(real_data, torch.Tensor):
        if real_data.size(axis) != 2:
            raise ValueError(f"Expected size-2 along axis {axis}, got shape {real_data.shape}")
        
        real = real_data.select(axis, 0)
        imag = real_data.select(axis, 1)
        return torch.complex(real, imag)

    else:
        raise NotImplementedError("Input must be a NumPy array or PyTorch tensor.")


# Metrics =====

def normalize(y, y_pred):
    """Normalization to preserve image contrast"""
    norm = y.max()
    y_norm = y / norm
    y_pred_norm = y_pred / norm
    return y_norm, y_pred_norm

def ssim_batch(y_batch, y_pred_batch):
    """Calculate ssim for a batch and return mean ssim."""
    all_ssim = []
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        y_norm, y_pred_norm = normalize(y, y_pred)
        all_ssim.append(ssim(y_norm, y_pred_norm))

    return all_ssim

def ssim(y, y_pred, data_range=1.0):
    """Normalize data and calculate ssim for one sample pair."""
    return structural_similarity(y, y_pred, data_range=data_range)

def psnr_batch(y_batch, y_pred_batch):
    """Calculate psnr for a batch and return mean psnr."""
    all_psnr = []
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        y_norm, y_pred_norm = normalize(y, y_pred)
        all_psnr.append(psnr(y_norm, y_pred_norm))

    return all_psnr

def psnr(y, y_pred, MAX_PIXEL_VALUE=1):
    """Calculate psnr for one sample pair."""
    rmse_ = rmse(y, y_pred)
    if rmse_ == 0:
        return float('inf')
    
    return 20 * math.log10(MAX_PIXEL_VALUE/rmse_+1e-10)

def mse(y, y_pred):
    """Calculate mean squared error."""
    return np.mean((y-y_pred)**2)

def rmse(y, y_pred):
    """Calculate root mean squared error."""
    return math.sqrt(mse(y, y_pred))