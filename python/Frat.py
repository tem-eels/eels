import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from ReadData import ReadData

def Frat(lfile=None, fwhm2=None, cfile=None):
    """
    Fourier-ratio deconvolution using exact method (A) with left shift before forward transform.
    Re-convolution function R(F) is exp(-X*X).
    Data is read in from named input files and output data appears in a named output file as nn x-y pairs.
    """

    print('\n----------------Frat---------------\n\n')

    if lfile is None:
        print("Alternate Usage: Frat('LLFile', fwhm, 'CLFile')\n")
        lfile = input('Name of low-loss file (e.g. CoreGen.low): ')
    else:
        print(f'Name of low-loss file: {lfile}')

    # Read low-loss data from input file
    data = ReadData(lfile, 2)

    nd = len(data)
    nn = 2 ** (int(np.log2(nd)) + 1)

    lldata = np.zeros(nn)
    e = np.zeros(nn)
    d = np.zeros(nn)

    e[:nd] = data[:, 0]
    lldata[:nd] = data[:, 1]

    epc = (e[4] - e[0]) / 4
    back = np.sum(lldata[:5])
    back += np.sum(lldata[:5]) / 5

    # Find zero-loss channel
    nz = np.argmax(lldata) + 1

    # Find minimum in J(E)/E to estimate zero-loss sum A0
    nsep = nz
    for i in range(nz, nd+1):
        if lldata[i] / (i-nz+1) > lldata[i-1] / (i-nz):
            break
        nsep = i
    
    sum_nsep = np.sum(lldata[:nsep])
    a0 = sum_nsep - back * nsep
    nfin = nd - nz + 1

    # Transfer shifted data to array d
    d[:nfin] = lldata[nz-1:nd] - back

    # Extrapolate the spectrum to zero at channel nn
    a1 = np.sum(d[nfin-10:nfin-5])
    a2 = np.sum(d[nfin-5:nfin])
    r = 2 * np.log((a1 + 0.2) / (a2 + 0.1)) / np.log((nd - nz) / (nd - nz - 10))
    dend = d[nfin-1] * ((nfin-1) / (nn-2-nz)) ** r

    cosb = 0.5 - 0.5 * np.cos(np.pi * np.arange(nn-nfin+1) / (nn-nfin-nz-1))
    d[nfin-1:] = d[nfin-1] * ((nfin-1) / np.arange(nfin-1, nn)) ** r - cosb * dend

    # Compute total area
    at = np.sum(d)

    # Add left half of Z(E) to end channels in the array d
    d[nn+2-nz-1:] = lldata[:nz-1] - back

    print(f'ND,NZ,NSEP,DATA(NZ): {nd}, {nz}, {nsep}, {lldata[nz-1]}')
    print(f'BACK,A0,DEND,EPC: {back}, {a0}, {dend}, {epc}')
    fwhm1 = 0.9394 * a0 / d[0] * epc
    print(f'zero-loss FWHM = {fwhm1:.1f} eV;\n')

    if cfile is None:
        fwhm2 = float(input('Energy resolution of coreloss SSD (e.g. FWHM above) = '))
        cfile = input('Name of coreloss file (e.g. CoreGen.cor): ')
    else:
        print(f'Energy resolution of coreloss SSD (FWHM) = {fwhm2}')
        print(f'Name of coreloss file: {cfile}')

    fwhm2 = fwhm2 / epc

    # Read core-loss data from input file
    data = ReadData(cfile, 2)
    nc = len(data)

    e[:nc] = data[:, 0]
    c = np.zeros(nn)
    c[:nc] = data[:, 1]

    # Extrapolate the spectrum to zero at channel nn
    a1 = np.sum(c[nc-10:nc-5])
    a2 = np.sum(c[nc-5:nc])
    r = 2 * np.log((a1 + 0.2) / (a2 + 0.1)) / np.log(e[nc-1] / e[nc-10])
    cend = a2 / 5 * (e[nc-3] / (e[0] + epc * (nn-1))) ** r
    cosb = 0.5 - 0.5 * np.cos(np.pi * np.arange(nn-nc+1) / (nn-nc))
    c[nc-1:] = a2 / 5 * (e[nc-3] / (e[0] + epc * np.arange(nc-3, nn-2))) ** r - cosb * cend

    # Write background-stripped core plural scattering distribution to Frat.psd
    eout = e[0] + epc * np.arange(-1, nn-1)
    cpout = np.real(c[:nn])
    np.savetxt('Frat.psd', np.column_stack((eout, cpout)), fmt='%8.15g')

    d = np.conj(fft(d, nn))
    c = np.conj(fft(c, nn))

    # Process the Fourier coefficients
    d = d + 1e-10
    c = c + 1e-10
    x = np.concatenate([np.arange(nn//2), np.arange(nn//2, 0, -1)])
    x = 1.887 * fwhm2 * x / nn
    gauss = a0 / np.exp(x ** 2) / nn
    d = gauss * c / d
    d = fft(d, nn)


    # Write SSD to output file
    csout = np.real(d[:nn])
    np.savetxt('Frat.ssd', np.column_stack((eout, csout)), fmt='%8.15g')

    # Plot
    plt.figure()
    plt.plot(eout, csout, 'r', label='SSD')
    plt.plot(eout, cpout, 'g', label='PSD')
    plt.legend()
    plt.title('Frat Output', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    Frat()