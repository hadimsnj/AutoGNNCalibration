import torch
import numpy as np
from scipy import stats
import csv
import math

def read_file_calibration(file_path):
    w = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            for r in row:
                try:
                    r = int(r)
                except ValueError:
                    r = float(r)
                    w.append(r)
    
    return w


def read_file_quantization(file_path):
    w = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            for r in row:
                try:
                    r = int(r)        
                except ValueError:
                    r = float(r)
                    
                w.append(r)
    
    return w
# ========================================================
# 
#                       MSE
# 
# ========================================================
def clamp(input, min, max, inplace=False):
    
    if inplace:
        input.clamp_(min, max)
        
        return input
    
    return torch.clamp(input, min, max)


def get_quantized_range(num_bits, signed=True):
    
    if signed:
        n = 2 ** (num_bits - 1)
        
        return -n, n - 1
    
    return 0, 2 ** num_bits - 1


def symmetric_linear_quantization_scale_factor(num_bits, saturation_val):
    
    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1
    
    return n / saturation_val


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    
    return input / scale_factor


def distiller_quantize(x, num_bits, alpha, signed):
    min_q_val, max_q_val = get_quantized_range(num_bits, signed=True)
    scale = symmetric_linear_quantization_scale_factor(num_bits, alpha)
    q = linear_quantize_clamp(torch.from_numpy(x), scale, min_q_val, max_q_val)
  
    x = linear_dequantize(q, scale)
    
    return x.numpy()


def mse_histogram_clip(bin_x, bin_y, num_bits, alpha, signed):
    idx = np.abs(bin_x) > alpha
    mse = np.sum((np.abs(bin_x[idx]) - alpha)**2 * bin_y[idx])

    idx = np.abs(bin_x) <= alpha
    bin_xq = distiller_quantize(bin_x[idx], num_bits, alpha, signed)
    mse += np.sum((bin_x[idx] - bin_xq)**2 * bin_y[idx])

    return mse


def mmse(values, num_bits, signed=True):
    
    values = np.array(values)
    if not np.isfinite(values).all():
        raise ValueError("Input to mmse contains NaN or infinite values.")
    
    max_abs = np.max(np.abs(values))
    if max_abs == 0 or not np.isfinite(max_abs):
        max_abs = 1e-6
    
    bin_y, bin_edges = np.histogram(values, bins=201, density=True)
    bin_x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_y /= np.sum(bin_y)
    
    alphas = np.arange(0.01, 1, 0.01) * max_abs
    mses = [mse_histogram_clip(bin_x, bin_y, num_bits, alpha, signed) for alpha in alphas ]
    
    alpha_best = alphas[np.argmin(mses)]
    # print(f"MSE alpha_best = {alpha_best:5.2f}  / {max_abs:5.2f}")
    
    return alpha_best





# ========================================================
# 
#                       KLD
# 
# ========================================================
def get_kld_threshold_15bins(arr):

    opt_th = _get_optimal_threshold(arr, num_bins=2001, num_quantized_bins=15)

    return opt_th



def _get_optimal_threshold(arr, num_bins=1001, num_quantized_bins=255):

    if not isinstance(arr, np.ndarray):
        raise TypeError('get_optimal_threshold only supports input type of np.ndarray,'
                        ' while received type=%s' % (str(type(arr))))
    
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2
    if not np.allclose(hist_edges[zero_bin_idx] + hist_edges[zero_bin_idx + 1],
                   0, rtol=1e-5, atol=1e-7):
        return max(abs(min_val), abs(max_val), 1e-6)

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    
    # i means the number of bins on half axis excluding the zero bin.
    for i in range(num_quantized_bins // 2, num_bins // 2 + 1):
        
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (sliced_nd_hist != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = p.size // num_quantized_bins
        
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        
        # expand quantized_bins into p.size bins
        q = np.zeros(p.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        
        q[sliced_nd_hist == 0] = 0
        p = _smooth_distribution(p)
        
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        else:
            divergence[i - num_half_quantized_bins] = stats.entropy(p, q)
        
        quantized_bins[:] = 0

    min_divergence_idx = np.argmin(divergence)
    opt_th = thresholds[min_divergence_idx]

    return opt_th
    

def _smooth_distribution(p, eps=0.0001):
   
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    
    return hist


def KLD_threshold(values, num_bits):
    values = np.array(values, dtype=np.float32)

    # remove NaN / Inf
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 1e-6

    max_abs = np.max(np.abs(values))
    if max_abs == 0 or not np.isfinite(max_abs):
        return 1e-6

    num_quantized_bins = 2**num_bits - 1
    opt_th = _get_optimal_threshold(values, num_quantized_bins=num_quantized_bins)

    if not np.isfinite(opt_th) or opt_th == 0:
        return 1e-6

    return float(opt_th)
    

    
def calculate_bits(number):
    
    integer_part = math.floor(number)

    if integer_part == 0:
        integer_bits = 1  # To represent 0, we still need at least one bit
    else:
        integer_bits = math.floor(math.log2(abs(integer_part))) + 1

    return integer_bits


def fake_quantization(data, calibration):
    
    for i in range(len(data)):
        if data[i] > calibration:
            data[i] = calibration
        elif data[i] < -calibration:
            data[i] = -calibration

    return data

# ========================================================
#
#                 Percentile Calibration
#
# ========================================================
def percentile_threshold(values, percentile=99.9, symmetric=True):
    
    values = np.array(values, dtype=np.float32)
    
    if symmetric:
        alpha_best = np.percentile(np.abs(values), percentile)
    else:
        lower = np.percentile(values, 100 - percentile)
        upper = np.percentile(values, percentile)
        alpha_best = max(abs(lower), abs(upper))
        
    if alpha_best == 0 or not np.isfinite(alpha_best):
        alpha_best = 1e-6
        
    return alpha_best
        
    