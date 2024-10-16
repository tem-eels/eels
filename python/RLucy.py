import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from ReadData import ReadData

def findLocalMax(in_array, window):
    lm = 0
    for k in range(len(in_array) - 2 * window):
        if np.sum(in_array[k+window:k+2*window]) < np.sum(in_array[k:k+window]):
            lm = np.argmax(in_array[k:k+2*window]) + k
            break
    if lm == 0:
        raise ValueError('findLocalMax: no maximum found')
    return lm

def RLucy(specFile=None, kernelFile=None, niter=None):
    """
    Performs Richardson-Lucy deconvolution on the given spectrum.
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011.
    """
    
    print("\n----------------RLucy---------------\n\n")
    
    if specFile is None or niter is None:
        print("Alternate Usage: RLucy('specFile', 'kernelFile', numIterations)\n\n")
        specFile = input('RLucy: Spectrum data file name (e.g. SpecGen.psd): ')
        kernelFile = input('Kernel file name (leave blank to generate from spectrum): ')
        niter = int(input('Number of iterations (e.g. 15): '))
    else:
        print(f'RLucy: Spectrum data file name: {specFile}')
        print(f'Kernel file name (generate from spectrum if blank): {kernelFile}')   
        print(f'Number of iterations: {niter}')
    
    # Read in spectrum file 
    data = ReadData(specFile, 2)
    
    nd = len(data)
    e = data[:, 0]
    spec = data[:, 1]
    
    back = np.mean(spec[:5])  # calculate background from first five data points
    
    # Find zero-loss channel:
    nzpre = np.argmax(spec > back * 5) + 1  # find where data starts to rise at least 5x 'back'
    nz = nzpre + findLocalMax(spec[nzpre-1:], 5)
    
    if not kernelFile:
        # Create kernel from first half of zero loss peak
        kernel = np.concatenate([spec[:nz], spec[:nz-1][::-1]])
    else:
        # Read in kernel file
        data = ReadData(kernelFile, 2)
        kernel = data[:, 1]
    
    ksize = len(kernel)  # kernel size
    
    # Pad kernel if it is smaller than spec
    if ksize < nd:
        kernel = np.pad(kernel, (0, nd - ksize), 'constant', constant_values=back)
    
    kernelOrig = kernel.copy()  # keep copy of original kernel
    # Shift zero loss peak to start
    ptm = np.roll(kernel, -nz+1)
    
    
    # Normalize kernel
    ptm = ptm / np.sum(ptm)
    kernel = fft(ptm)
    
    # Set starting spectrum (orr) as spec
    orr = spec
    
    # Set up plk
    ptm = fft(orr)
    ptm = ptm * kernel
    plk = ifft(ptm)
    
    fdk = 0
    # Start deconvolution
    for iter in range(niter):
        fnum = spec + fdk
        fden = fdk
        
        fden += plk
        ptm = fnum.copy()
        ptm[fden != 0] = ptm[fden != 0] / fden[fden != 0]
        
        plk = fft(ptm)
        ptm = kernel
        ptm = ptm * plk
        plk = ifft(ptm)
        orr = orr * plk.real
        ptm = fft(orr)
        ptm = ptm * kernel
        plk = ifft(ptm)
    
    # Plot results
    plt.figure()
    plt.plot(e, kernelOrig, label='Original Kernel')
    plt.title('RLucy Kernel')
    plt.ylabel('Counts')
    plt.show(block=False)

    plt.figure()
    plt.plot(e, spec, 'g', label='Original Spectrum')
    plt.plot(e, orr.real, 'b', label='Sharpened Spectrum')
    plt.title('RLucy Output')
    plt.legend()
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('Counts')
    plt.show()

# Example usage:
# RLucy('SpecGen.psd', '', 15)
if __name__ == '__main__':
    RLucy()