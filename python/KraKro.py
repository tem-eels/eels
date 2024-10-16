import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from ReadData import ReadData

def KraKro(infile=None, a0=None, e0=None, beta=None, ri=None, nloops=None, delta=None):
    """
    Kramers-Kronig analysis using Johnson method 
    Program generates output into the 5-column file KRAKRO.DAT.
    """

    print("\n--------------KraKro-----------------\n\n")

    if infile is None:
        print("Alternate Usage: KraKro('InFile', A0, E0, Beta, RefIndex, Iterations, delta)\n\n")
        infile = input("Name of input file (e.g. Drude.ssd): ")
    else:
        print(f"Name of input file: {infile}")
    
    data = ReadData(infile, 2)

    if any(param is None for param in [a0, e0, beta, ri, nloops, delta]):
        a0 = float(input('Zero-loss sum (1 for Drude.ssd): '))
        e0 = float(input('E0(keV): '))
        beta = float(input('BETA(mrad): '))
        ri = float(input('ref.index: '))
        nloops = int(input('no. of iterations: '))
        delta = float(input('stability parameter (0.1 - 0.5 eV): '))
    else:
        print(f'Zero-loss sum (1 for Drude.ssd): {a0}')
        print(f'E0(keV): {e0}')
        print(f'BETA(mrad): {beta}')
        print(f'ref.index: {ri}')
        print(f'no. of iterations: {nloops}')
        print(f'stability parameter (0.1 - 0.5 eV): {delta}')

    # Calculate NN based on size of input file
    dsize = len(data)
    nn = 2 ** (int(np.log2(dsize)) + 1) * 4
    nlines = dsize

    # Assign data to SSD and pad unused space with zeroes
    en = np.zeros(nn)
    ssd = np.zeros(nn)
    en[1:dsize+1] = data[:, 0]
    ssd[1:dsize+1] = data[:, 1]
    d = ssd.copy()  # single-scattering distribution (intensity or probability)
    epc = (en[4] - en[0]) / 4  # eV/channel

    t = e0 * (1 + e0 / 1022.12) / (1 + e0 / 511.06) ** 2  # t = mv^2/2
    rk0 = 2590 * (1 + e0 / 511.06) * np.sqrt(2 * t / 511.06)  # k0 (wavenumber)
    tgt = e0 * (1022.12 + e0) / (511.06 + e0)  # 2.gamma.t
    e = epc * np.arange(1, nn)  # energy-loss values
    print(f' nlines= {nlines} , nn = {nn}, epc = {epc}')

    # Calculate Im(-1/EPS), Re(1/EPS), EPS1, EPS2 and SRFINT data:
    for num in range(nloops):
        # Apply aperture correction APC at each energy loss:
        area = np.sum(d[1:nn]) * epc  # integral of [(dP/dE).dE]
        apc = np.log(1 + (beta * tgt / e) ** 2)
        d[1:nn] = d[1:nn] / apc
        dsum = np.sum(d[1:nn] / e) * epc  # sum of counts/energy
        
        rk = dsum / 1.571 / (1 - 1 / ri ** 2)  # normalization factor
        tnm = 332.5 * rk / a0 * e0 * (1 + e0 / 1022.12) / (1 + e0 / 511.06) ** 2
        tol = area / a0
        rlam = tnm / tol
        print(f' LOOP {num + 1} : t(nm) = {np.real(tnm)} , t/lambda = {np.real(tol)} lambda(nm) = {np.real(rlam)}')

        d = d / rk  # Im(-1/eps)
        imreps = d.copy()  # stored value, not to be transformed
        
        d = fft(d, nn)  # Fourier transform   
        d = -2 * np.imag(d) / nn  # Transfer sine coefficients to cosine locations:
        d[:nn//2] = -d[:nn//2]        
        d = fft(d, nn)  # Inverse transform
        
        # Correct the even function for reflected tail:
        dmid = np.real(d[nn//2-1])
        d[:nn//2] = np.real(d[:nn//2]) + 1 - dmid / 2 * ((nn/2) / np.arange(nn-1, nn//2-1, -1)) ** 2
        d[nn//2:] = 1 + dmid * ((nn/2) / np.arange(nn//2+1, nn+1)) ** 2 / 2

        

        re = np.real(d[1:nn]).copy()
        den = re * re + imreps[1:nn] * imreps[1:nn]     
        eps1 = re / den
        eps2 = imreps[1:nn] / den
        
        # Calculate surface energy-loss function and surface intensity:
        srfelf = 4 * eps2 / ((1 + eps1) ** 2 + eps2 ** 2) - imreps[1:nn]
        adep = tgt / (e + delta) * np.arctan(beta * tgt / e) - beta / 1000 / (beta ** 2 + e ** 2 / tgt ** 2)
        srfint = 2000 * rk / rk0 / tnm * adep * srfelf
        d[1:nn] = ssd[1:nn] - srfint  # correct ssd for surface-loss intensity
        d[0] = 0

    # Limit data to requested NLINES
    e = np.real(e[:nlines])
    eps1 = np.real(eps1[:nlines])
    eps2 = np.real(eps2[:nlines])
    re = np.real(re[:nlines])
    imreps = np.real(imreps[1:nlines+1])  # Mike's change
    srfelf = np.real(srfelf[:nlines]) 
    srfint = np.real(srfint[:nlines])
    ssd = np.real(ssd[:nlines])

    # Write data to KraKro.dat
    np.savetxt('KraKro.dat', np.column_stack((e, eps1, eps2, re, imreps)), fmt='%f')

    # Plot EPS1, EPS2, RE, DI
    plt.figure()
    plt.plot(e, imreps, 'k', label='Imag(-1/eps)')
    plt.plot(e, re, 'm', label='Real(1/eps)')
    plt.plot(e, eps1, 'r', label='eps1')
    plt.plot(e, eps2, 'g', label='eps2')
    plt.plot(e, srfelf, 'b', label='Im(-4/1+eps)')
    plt.legend()
    plt.title('KraKro', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    
    # Limit the y-scale of plot to +/- 2* the elf peak
    yScaleMax = np.max(imreps) * 2
    plt.ylim([-yScaleMax, yScaleMax])
    plt.show(block=False)

    # Plot ssd, srfint
    plt.figure()
    plt.plot(e, ssd, 'k', label='ssd')
    plt.plot(e, srfint, 'm', label='srfint')
    plt.legend()
    plt.title('KraKro intensities', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.show()

if __name__ == "__main__":
    KraKro()