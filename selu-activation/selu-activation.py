import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    list=[]
    for i in x:
        if(i>0):
            list.append(lam*i)
        else:
            list.append(lam*alpha*(np.exp(i)-1))
    # Write code here
    return list
    pass
