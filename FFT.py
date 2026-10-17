import numpy as np

""" SO this is an example code for an FFT, I will maybe also add another function for the bit manipulation
 one that's a fuck ton times more expensive, but oh well, right now it's just a very cool FFT function that
 that can also reverse.
"""
def FFT(Coeff: np.array, reverse: bool) -> np.array:
    #base case:
    if (Coeff.size == 1):
        return Coeff

    # split into even and odd indices
    P_e = Coeff[::2]
    P_o = Coeff[1::2]
    #P_e and P_o put back into the recursive function, returns an array that is the Polynomial evaluatd at specific roots of unity
    P_eE = FFT(P_e, reverse)
    P_oE = FFT(P_o, reverse)
    PE = np.empty(Coeff.size, dtype=complex)
    if (reverse):
        for k in range(0, Coeff.size//2):
            PE[k] = P_eE[k] +  np.exp(-2*np.pi*1j*k/Coeff.size)*P_oE[k]
            PE[k + Coeff.size/2] = P_eE[k] -  np.exp(-2*np.pi*1j*k/Coeff.size)*P_oE[k]
        return PE/2
    else:
        for k in range(0, Coeff.size//2):
            PE[k] = P_eE[k] +  np.exp(2*np.pi*1j*k/Coeff.size)*P_oE[k]
            PE[k + Coeff.size/2] = P_eE[k] -  np.exp(2*np.pi*1j*k/Coeff.size)*P_oE[k]
        return PE

        
