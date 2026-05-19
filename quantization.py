import torch


# =========================================================
#        Quantization parameter generation
# =========================================================
def generate_quantization_constants(alpha, beta, alpha_q, beta_q):
    s = (beta - alpha) / (beta_q - alpha_q)
    z = (beta * alpha_q - alpha * beta_q) / (beta - alpha)
    return s, z


def generate_quantization_uqbits_constants(alpha, beta, qbits):
    alpha_q = 0
    beta_q = (2 ** qbits - 1)

    s, z = generate_quantization_constants(
        alpha=alpha,
        beta=beta,
        alpha_q=alpha_q,
        beta_q=beta_q,
    )
    return s, z


def generate_quantization_qbits_constants(alpha, beta, qbits):
    if qbits == 1:
        alpha_q = -1
        beta_q = 1
    else:
        alpha_q = -2 ** (qbits - 1) + 1
        beta_q = 2 ** (qbits - 1) - 1

    s, z = generate_quantization_constants(
        alpha=alpha,
        beta=beta,
        alpha_q=alpha_q,
        beta_q=beta_q,
    )
    return s, z


# =========================================================
#                Fake Quantization
# =========================================================
def fake_quantization(x, s, z, alpha_q, beta_q):
    x_int = torch.round(x / s + z)
    x_int = torch.clamp(x_int, alpha_q, beta_q)
    x_dequant = (x_int - z) * s
    return x_dequant


def fake_quantization_b(x, s, z, alpha_q, beta_q):
    x_int = torch.round(x / s + z)
    x_int = torch.clamp(x_int, alpha_q, beta_q)
    x_dequant = (x_int - z) * s
    return x_dequant


# =========================================================
#                Quantization APIs
# =========================================================
def quantization_fbits(x, s, z, qbits, ste=False):
    if qbits == 1:
        alpha_q = -1
        beta_q = 1
        x_q = fake_quantization_b(x, s, z, alpha_q, beta_q)
    else:
        alpha_q = -2 ** (qbits - 1) + 1
        beta_q = 2 ** (qbits - 1) - 1
        x_q = fake_quantization(x, s, z, alpha_q, beta_q)

    if ste:
        return x + (x_q - x).detach()
    return x_q


def quantization_ufbits(x, s, z, qbits, ste=False):
    alpha_q = 0
    beta_q = 2 ** qbits - 1

    x_q = fake_quantization(x, s, z, alpha_q, beta_q)

    if ste:
        return x + (x_q - x).detach()
    return x_q