import numpy as np
import matplotlib.pyplot as plt

def tKKs(inFile=None, E0=None, n=None, beta=None, I0=None, zlpMethod=None):
    """
    Absolute Thickness from the Kramers-Kronig Sum Rule.
    Assumes collection semi-angle << (Ep/E0)^0.5 (dipole scattering).
    The zero-loss peak is fitted to a Gaussian function and subtracted.
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011.
    Version 10.11.26, surface-mode correction delta_t set to zero.
    """
    
    print("tKKs( 'inFile', E0, RefIdx, Beta, I0, 'zlpMethod')\n")
    
    if inFile is None or E0 is None or n is None or beta is None or I0 is None:
        inFile = input('Input data file: ')
        E0 = float(input('E0 (keV): '))
        n = float(input('Refractive Index (for metal/semimetal enter 0): '))
        beta = float(input('Collection semiangle (mrad): '))
        I0 = float(input('Zero-loss integral (enter 0 if spectrum has a ZLP): '))
        if I0 == 0:
            zlpMethod = input('Enter ZLP Method (Gaussian, Actual, LeftSide, Smoothed): ')
    else:
        print(f'Input data file: {inFile}')
        print(f'E0 (keV): {E0}')
        print(f'Refractive Index (For metal/semimetal enter 0): {n}')
        print(f'Collection angle (mrad): {beta}')
        print(f'Zero-loss integral (0 if spectrum has a ZLP): {I0}')
        if I0 == 0:
            print(f'ZLP Method: {zlpMethod}')
    
    if n == 0:  # if metal/semimetal set n = 1000
        n = 1000
    
    thetar = 1000. * (50. / E0 / 1000.) ** 0.5  # Bethe-ridge angle (mrad) for E=50 eV
    if beta > thetar:  # approximate treatment for large collection angle 
        beta = thetar
        print('beta replaced by 50eV Bethe-ridge angle\n')
    
    E, spec = GetSpectrum(inFile)  # read in spectrum from file
    
    back = np.mean(spec[:5])
    eVperChan = E[1] - E[0]  # eV per channel
    
    if I0 == 0:  # calculate ZLP integral if input parameter = 0
        zlpChan = findZeroLossPeak(spec)
        zlpFWHM = calcFWHM(E, spec, zlpChan)
        spec = spec - back  # background needs to be removed after finding ZLP
        
        if zlpMethod.lower() == 'gaussian':
            sigma = zlpFWHM / 2.35482
            zlp = (spec[zlpChan] - back) * np.exp(-((E - E[zlpChan]) ** 2) / (2. * sigma ** 2))
        
        elif zlpMethod.lower() == 'actual':
            zlpRightChan = zlpChan - 1 + findLocalMax(-spec[zlpChan:], 5)
            zlp = np.where(E <= E[zlpRightChan], spec, 0)
        
        elif zlpMethod.lower() == 'leftside':
            zlp = np.zeros_like(spec)
            left_side = spec[:zlpChan]
            zlp[:2 * zlpChan - 1] = np.concatenate([left_side, left_side[-2::-1]])
        
        elif zlpMethod.lower() == 'smoothed':
            zlpRightChan = zlpChan - 1 + findLocalMax(-spec[zlpChan:], 5)
            zlp = np.where(E <= E[zlpRightChan], spec, 0)
            zlp = zlp - np.where((E > 0) & (E <= E[zlpRightChan]), spec[zlpRightChan] * E / E[zlpRightChan], 0)
        else:
            raise ValueError('ZLP reconstruction method unknown')
        
        specInelas = spec - zlp  # remove zlp from spectrum
        
        plt.figure()
        plt.plot(E, zlp, 'b', label='ZLP')
        plt.plot(E, specInelas, 'g', label='Inelastic')
        plt.legend()
        plt.title('tKKs input data', fontsize=12)
        plt.xlabel('Energy Loss [eV]')
        plt.tight_layout()
        plt.show()
        
        I0 = np.trapz(zlp, dx=eVperChan)  # zero-loss integral
        Ii = np.trapz(specInelas, dx=eVperChan)  # inelastic integral
        It = I0 + Ii  # total integral
        
        # Plural-scattering correction
        log_ratio = np.log(It / I0)
        C = 1 + log_ratio / 4 + (log_ratio) ** 2 / 18 + (log_ratio) ** 3 / 96
        
    else:  # if ZLP is missing and I0 is supplied as an input number
        spec = spec - back 
        specInelas = spec
        Ii = np.trapz(spec, dx=eVperChan)  # spectrum sum
        C = 1  # assume that a spectrum with no ZLP is a SSD
    
    delta_t = 0  # surface correction in nm
    a0 = 0.0529  # in nm
    T = E0 * (1 + E0 / 1022) / (1 + E0 / 511) ** 2
    thetaE = (E / E0) * (E0 + 511) / (E0 + 1022)  # in mrad
    
    tCoeff = 2000. * a0 * T / (C * I0 * (1 - n ** -2))  # eV
    with np.errstate(divide='ignore', invalid='ignore'):
        tInt = np.where(E > 0, specInelas / (E * np.log(beta / thetaE)), 0)  # integrand, eV^-2
    tInt = np.nan_to_num(tInt, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate thickness
    t = tCoeff * np.trapz(tInt, dx=eVperChan) - delta_t  # Eq.(5.9)
    
    print(f'specimen thickness (nm) = {t}')
    if C == 1:  # SSD with no ZLP
        print(f't/IMFP = {Ii / I0}')
        print(f'IMFP (nm) = {t / (Ii / I0)}')
    else:  # spectrum with ZLP and plural scattering
        print(f't/IMFP = {np.log(It / I0)}')
        print(f'IMFP (nm) = {t / np.log(It / I0)}')
    print('-------------------------------\n')

def GetSpectrum(inFile):
    """
    General function to read in spectrum data from a data file.
    Function assumes 2 column input file with energy and intensity.
    """
    try:
        data = np.loadtxt(inFile)
        E = data[:, 0]
        spec = data[:, 1]
        
        # Replace actual zero value with near zero
        Ezero_indices = np.where(E == 0)[0]
        if Ezero_indices.size > 0:
            E[Ezero_indices] = 1e-5
        return E, spec
    except Exception as e:
        raise IOError(f"Error reading file {inFile}: {e}")

def findZeroLossPeak(spec):
    """
    Returns zero loss peak channel from a spectrum.
    """
    back = np.mean(spec[:5])  # calculate background from first five data points
    nzpre = np.argmax(spec > back * 5)  # find where data starts to rise at least 5x 'back'
    if nzpre == 0 and spec[0] <= back * 5:
        nzpre = 0
    zlpChan = nzpre + findLocalMax(spec[nzpre:], 5)
    return zlpChan

def calcFWHM(E, spec, peakChan):
    """
    Calculates FWHM from a spectrum and the location of a peak.
    """
    halfMax = spec[peakChan] / 2
    # Left side
    left_indices = np.where(spec[:peakChan] <= halfMax)[0]
    if left_indices.size == 0:
        x1 = 0
    else:
        x1 = left_indices[-1]
    
    # Right side
    right_indices = np.where(spec[peakChan:] <= halfMax)[0]
    if right_indices.size == 0:
        x2 = len(spec) - 1
    else:
        x2 = peakChan + right_indices[0]
    
    fwhm = E[x2] - E[x1]
    return fwhm

def findLocalMax(arr, window):
    """
    Finds local maximum in an array within a specified window.
    """
    for k in range(len(arr) - 2 * window + 1):
        if np.sum(arr[k + window:k + 2 * window]) <= np.sum(arr[k:k + window]):
            lm = np.argmax(arr[k:k + 2 * window]) + k
            return lm
    return np.argmax(arr)  # Fallback to the global maximum

if __name__ == "__main__":
    tKKs()