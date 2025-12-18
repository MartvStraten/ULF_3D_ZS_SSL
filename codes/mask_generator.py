import numpy as np
import sigpy.mri as spmri


def zpad(array, new_shape):
    """
    Zero-pad the input array to the specified new shape.

    Parameters:
    array (numpy.ndarray): The input array to be padded.
    new_shape (tuple): The desired shape of the output array (height, width).

    Returns:
    numpy.ndarray: The zero-padded array with the specified new shape.
    """
    """Zero-pad the input array to the specified new shape."""
    pad_height = new_shape[0] - array.shape[0]
    pad_width = new_shape[1] - array.shape[1]
    return np.pad(array, ((pad_height // 2 , pad_height - pad_height // 2  ),
                           (pad_width // 2 , pad_width - pad_width // 2 )),
                  mode='constant', constant_values=0)


def get_calib_size(mask):
    """
    Determine the size of the largest rectangle in the middle of the binary mask,
    which expands in 4 directions one step at a time starting from a 1x1 region in the center.

    Args:
        mask: 2D binary numpy array.

    Returns:
        calib_size: Array representing the size of the largest rectangle (height, width).
    """
    # Find the center of the mask
    center_x = mask.shape[0] // 2
    center_y = mask.shape[1] // 2
    
    # Initialize the rectangle with a size of 1x1 centered in the mask
    top, bottom = center_x, center_x
    left, right = center_y, center_y
    
    # Flags to determine when we can't expand further in a given direction
    can_expand_up = True
    can_expand_down = True
    can_expand_left = True
    can_expand_right = True
    
    while can_expand_up or can_expand_down or can_expand_left or can_expand_right:
        # Try expanding upwards
        if can_expand_up and top > 0 and np.all(mask[top-1, left:right+1]):
            top -= 1
        else:
            can_expand_up = False

        # Try expanding downwards
        if can_expand_down and bottom < mask.shape[0] - 1 and np.all(mask[bottom+1, left:right+1]):
            bottom += 1
        else:
            can_expand_down = False

        # Try expanding leftwards
        if can_expand_left and left > 0 and np.all(mask[top:bottom+1, left-1]):
            left -= 1
        else:
            can_expand_left = False

        # Try expanding rightwards
        if can_expand_right and right < mask.shape[1] - 1 and np.all(mask[top:bottom+1, right+1]):
            right += 1
        else:
            can_expand_right = False
    
    # Calculate the width and height of the rectangle
    height = bottom - top + 1
    width = right - left + 1
    
    calib_size = [height, width]
    
    return calib_size

    # Example usage:
    # mask = np.array([[0, 0, 0, 0, 0], 
    #                  [0, 1, 1, 1, 0],
    #                  [0, 1, 1, 1, 0],
    #                  [0, 1, 1, 1, 0],
    #                  [0, 0, 0, 0, 0]])  # Replace with actual mask data

    # calib_size = get_calib_size(mask)
    # print(calib_size)


def generate_poisson_mask(kx,ky,accelRate,cRP):
    """
    Generates a 2D Poisson-disc undersampling mask for MRI reconstruction.
    Parameters:
    kx (int): The number of frequency encoding steps.
    ky (int): The number of phase encoding steps.
    accelRate (float): The desired acceleration rate.
    cRP (float): The calibration region percentage.
    Returns:
    numpy.ndarray: A 2D array representing the undersampling mask.
    """
    calibRegion = [round(cRP*kx),round(cRP*ky)]

    # print(f"Calibration Size: {calibRegion}")
    numberOfDesiredPoints = (kx*ky)/accelRate # This would include calibration region
    r1 = calibRegion[0]/kx
    r2 = calibRegion[1]/ky
    numberOfNewPoints = numberOfDesiredPoints - np.prod(calibRegion)*(1/(1+r1))*(1/(1+r2))
    newAccelerationRate = (kx*ky)/numberOfNewPoints
    # print(f"New Acceleration Rate: {newAccelerationRate}")

    maskPois = spmri.poisson([kx,ky],newAccelerationRate,[0,0],seed=np.random.randint(1024),crop_corner=False,max_attempts=30) 

    # Enforce calibration region
    zz = zpad(np.ones(calibRegion), [kx,ky])
    # Logical or-gate maskPois and zz
    maskPois = np.logical_or(maskPois, zz)

    accelRatePois = np.prod(maskPois.shape)/np.sum(maskPois)
    # print(f"Final Acceleration Rate: {accelRatePois}")

    return maskPois,accelRatePois


def gen_pdf(sx,sy, p, accel, dist_type=2, radius=0):
    """
    Generates a probability density function (PDF) for a 1D 
    or 2D random sampling pattern with polynomial variable density.

    Parameters:
    -----------
    sx & sy : int
        Size of the matrix (2D).
    p : float
        Power of the polynomial for density control.
    accel : float
        Acceleration factor (e.g., 2 for 2x acceleration).
    dist_type : int, optional, default=2
        Distance type: 1 for L1 distance, 2 for L2 distance.
    radius : float, optional, default=0
        Radius for the fully sampled center region.

    Returns:
    --------
    pdf : numpy.ndarray
        The generated probability density function for sampling.
    """

    # Initialize values for bisection method to control sampling density
    minval, maxval = 0, 1
    val = 0.5

    # Handle cases where dimensions are zero
    sxx = 1 if sx == 0 else sx
    syy = 1 if sy == 0 else sy
    PCTG = int(np.floor((1 / accel) * sxx * syy))  # Target total number of samples


    # Generate coordinates and distance measures based on 1D or 2D input
    if sx !=  0 and sy != 0: # 2D
        x, y = np.meshgrid(np.linspace(-1, 1, sx), np.linspace(-1, 1, sy), indexing='xy')
        if dist_type == 1:
            r = np.maximum(np.abs(x), np.abs(y))  # L1 distance
        else:
            r = np.sqrt(x ** 2 + y ** 2)  # L2 distance
            r = r / np.max(np.abs(r))  # Normalize to range [0, 1]
    else:  # 1D
        r = np.abs(np.linspace(-1, 1, max(sx, sy)))

    # Set PDF values within the fully sampled center radius to 1
    idx = r < radius
    pdf = (1 - r) ** p
    pdf[idx] = 1

    # Ensure sampling criteria is feasible
    if np.floor(np.sum(pdf)) > PCTG:
        raise ValueError("Infeasible without undersampling DC, increase p")

    # Bisection method to adjust val to achieve the target sampling factor
    while True:
        val = (minval + maxval) / 2
        pdf = (1 - r) ** p + val
        pdf[pdf > 1] = 1
        pdf[idx] = 1  # Ensure central radius remains fully sampled

        N = int(np.floor(np.sum(pdf)))
        if N > PCTG:  # Infeasible, reduce upper bound
            maxval = val
        elif N < PCTG:  # Feasible but not optimal, increase lower bound
            minval = val
        else:  # Target sampling achieved, exit
            break

    return pdf

def gen_sampling(pdf, iterations, tolerance=None):
    """
    Generates a sampling pattern using a Monte Carlo algorithm with minimum peak interference.

    Parameters:
    -----------
    pdf : numpy.ndarray
        Probability density function from which to choose samples.
    iterations : int
        Number of tries for generating the sampling pattern.
    tolerance : int
        Allowed deviation from the desired number of samples.
        Default is 1% of the total number of elements in pdf.

    Returns:
    --------
    mask : numpy.ndarray
        Sampling pattern with minimum peak interference.
    stat : list of float
        List of minimum interference values for each iteration.
    act_pctg : float
        Actual undersampling factor.
    """
    # Set default tolerance to 1% of pdf elements if not provided
    if tolerance is None:
        tolerance = int(np.prod(pdf.shape) * 0.01)
    
    # Clamp any values in pdf greater than 1
    pdf = np.clip(pdf, 0, 1)
    K = np.sum(pdf)  # Target number of samples based on pdf

    # Initialize variables to track minimum interference
    min_intr = 1e9
    mask = np.zeros_like(pdf)
    stat = []

    # Monte Carlo sampling iterations
    for _ in range(iterations):
        tmp = np.zeros_like(pdf)
        
        # Generate a sampling pattern that meets the tolerance condition
        while abs(np.sum(tmp) - K) > tolerance:
            tmp = (np.random.rand(*pdf.shape) < pdf).astype(int)

        # Compute interference via inverse FFT and ignore DC component
        TMP = np.fft.ifft2(tmp / np.where(pdf == 0, 1, pdf))  # Avoid divide-by-zero
        TMP_magnitude = np.abs(TMP)
        
        # Ignore the DC component (first element) when evaluating interference
        max_interference = np.max(TMP_magnitude[1:])
        
        # Update minimum interference and corresponding sampling pattern if improved
        if max_interference < min_intr:
            min_intr = max_interference
            mask = tmp

        # Record interference for the current iteration
        stat.append(max_interference)
        
    # Calculate actual undersampling factor
    act_pctg = np.sum(mask) / np.prod(mask.shape)

    return mask, act_pctg, stat

def generate_pdf_mask(sx,sy, accel, p=2, dist_type=2, radius=0, iterations=1000, tolerance=None):
    """
    Generates a sampling mask using a polynomial-based PDF and Monte Carlo sampling.

    Parameters:
    -----------
    sx, sy : int
        Size of the sampling matrix in x and y dimensions.
    accel : float
        Target acceleration factor (e.g., 2 for 2x acceleration).
    p : int, optional (default=2)
        Initial polynomial power for density control. Increases iteratively if the PDF is infeasible.
    dist_type : int, optional (default=2)
        Distance metric: 1 for L1 distance, 2 for L2 distance.
    radius : float, optional (default=0)
        Radius for the fully sampled central region in the PDF.
    iterations : int, optional (default=1000)
        Number of Monte Carlo iterations to generate a sampling mask.
    tolerance : int, optional
        Allowed deviation from the target number of samples. Default is 1% of the total elements in pdf.

    Returns:
    --------
    mask : numpy.ndarray
        The generated sampling mask with minimal interference.
    act_pctg : float
        The actual undersampling factor achieved.
    """

    # Initialize generation flag
    gf = True

    # Try generating a feasible PDF, increasing polynomial order `p` if needed
    while gf:
        try:
            pdf = gen_pdf(sx,sy, p, accel, dist_type, radius)
            gf = False
        except ValueError:
            p += 1

    mask, act_pctg, _ = gen_sampling(pdf, iterations, tolerance)

    return mask, act_pctg
