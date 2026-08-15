import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """

    mu=np.mean(x, axis=-1, keepdims=True)
    sigma=np.var(x, axis=-1, keepdims=True)
    ln1=gamma*(x-mu)/sigma+beta

    numerator=x-mu
    denominator=np.sqrt(sigma+eps)
    ln=gamma*numerator/denominator+beta

    return ln
    # Your code here
    pass