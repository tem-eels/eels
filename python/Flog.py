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

def Flog(infile=None, fwhm2=None):
    """
    FOURIER-LOG DECONVOLUTION USING EXACT METHODS (A) OR (B)
    Single-scattering distribution is written to the file FLOG.DAT
    """
    print("\n----------------Flog---------------\n\n")

    # Read spectrum from input file
    if infile is None:
        print("Alternate Usage: Flog('Filename', FWHM2)\n\n")
        infile = input("Name of input file (e.g. SpecGen.psd) = ")
    else:
        print(f"Name of input file = {infile}")
    
    data = ReadData(infile, 2)
    nd = len(data)
    e = data[:, 0]
    psd = data[:, 1]
    epc = (e[4] - e[0]) / 4
    back = np.mean(psd[:5])

    # Calculate nn based on size of input file and round up to next power of 2
    nn = 2 ** int(np.ceil(np.log2(nd)))
    d = np.zeros(nn)
    z = np.zeros(nn)

    # Find zero-loss channel
    nzpre = np.argmax(psd > back * 5) + 1
    nz = nzpre + findLocalMax(psd[nzpre-1:], 5)

    # Find minimum in J(E)/E to separate zero-loss peak
    nsep = nz + findLocalMax(-psd[nz-1:], 5)

    dsum = np.sum(psd[:nsep])
    a0 = dsum - back * (nsep)
    nsep2 = nsep - nz + 1
    nfin = nd - nz + 1

    # Transfer shifted data to array d
    d[:nfin] = psd[nz-1:] - back

    # Extrapolate the spectrum to zero at end of array
    a1 = np.sum(d[nfin-10:nfin-5])
    a2 = np.sum(d[nfin-5:nfin])
    r = 2 * np.log((a1 + 0.2) / (a2 + 0.1)) / np.log((nd - nz) / (nd - nz - 10))
    if r <= 0:
        r = 0
    dext = d[nfin-1] * ((nfin-1) / (nn-2-nz)) ** r

    cosb = 0.5 - 0.5 * np.cos(np.pi * np.arange(nn-nfin+1) / (nn-nfin-nz-1))
    d[nfin-1:nn] = d[nfin-1] * ((nfin-1) / np.arange(nfin-1, nn)) ** r - cosb * dext

    # Copy zero-loss peak and smooth right-hand end
    z[:nsep2] = d[:nsep2] - d[nsep2-1] / 2 * (1 - np.cos(np.pi / 2 * np.arange(nsep2) / (nsep2-1)))
    z[nsep2-1:2*nsep2-1] = d[nsep2-1] / 2 * (1 - np.cos(np.pi / 2 * np.arange(nsep2-1, -1, -1) / (nsep2-1)))
    # Add left half of Z(E) to end channels of arrays D and Z
    d[-nz+1:] = psd[:nz-1] - back
    z[-nz+1:] = psd[:nz-1] - back

    print(f"NreaD, NZ, NSEP, BACK = {nd} {nz} {nsep} {back}")
    print(f"eV/channel = {epc}, zero-loss intensity = {a0}")
    fwhm1 = 0.9394 * a0 / d[0]
    print(f"FWHM = {fwhm1} channels.")

    if fwhm2 is None:
        fwhm2 = float(input("Enter new FWHM or 0 to keep same ZLP: "))
    else:
        print(f"New FWHM (0 to keep same ZLP): {fwhm2}")

    # Compute Fourier transforms
    z = np.conj(fft(z, nn))
    d = np.conj(fft(d, nn))

    # Process the Fourier coefficients
    d += 1e-10
    z += 1e-10
    dbyz = np.log(d / z) / nn  # /nn for correct scaling
    if fwhm2 == 0:  # use ZLP as reconvolution function
        d = z * dbyz
    else:
        x = np.concatenate([np.arange(nn//2), np.arange(nn//2, 0, -1)])
        x = 1.887 * fwhm2 * x / nn
        gauss = (x <= 9.0) * np.exp(-x * x)
        d = dbyz * gauss * a0

    d = fft(d, nn)

    # Write data to Flog.ssd
    eout = epc * np.arange(nz, nn + nz)
    dout = np.real(d)
    np.savetxt('Flog.ssd', np.column_stack((eout, dout)), fmt='%8.15g %8.15g')

    # Plot SSD and PSD
    plt.figure()
    plt.plot(eout + e[0], dout, 'b', label='SSD')
    plt.plot(e, psd, ':r', label='PSD')
    plt.legend()
    plt.title('FLog deconvolution', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('Count/channel')
    plt.show()


if __name__ == "__main__":
    Flog()