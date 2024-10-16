import numpy as np
import matplotlib.pyplot as plt
from ReadData import ReadData

def Afit(infile=None, eprestart=None, eprewidth=None, epoststart=None, epostwidth=None, outcore=None, outback=None):
    print('\n---------------Afit--------------\n\n')

    if infile is None:
        infile = input('Input file: ')
    else:
        print(f'Input file: {infile}')

    spectrum = ReadData(infile, 2).T

    print(f'\nEnergy dispersion = {spectrum[0][2] - spectrum[0][1]} [eV]\n')

    if eprestart is None or eprewidth is None or epoststart is None or epostwidth is None:
        eprestart = float(input('Pre-Edge energy window START [eV]: '))
        eprewidth = float(input('Pre-Edge energy window WIDTH [eV]: '))
        epoststart = float(input('Post-Edge energy window START [eV]: '))
        epostwidth = float(input('Post-Edge energy window WIDTH [eV]: '))
    else:
        print(f'Pre-Edge energy window START [eV]: {eprestart}')
        print(f'Pre-Edge energy window WIDTH [eV]: {eprewidth}')
        print(f'Post-Edge energy window START [eV]: {epoststart}')
        print(f'Post-Edge energy window WIDTH [eV]: {epostwidth}')

    efilter = ((spectrum[0] >= eprestart) & (spectrum[0] <= (eprestart + eprewidth))) | ((spectrum[0] >= epoststart) & (spectrum[0] <= (epoststart + epostwidth)))
    lspectrum = np.log(spectrum)
    sumx = np.sum(lspectrum[0][efilter])
    sumxovery = np.sum(lspectrum[0][efilter] / lspectrum[1][efilter])
    sumoneovery = np.sum(1 / lspectrum[1][efilter])
    sumxsqovery = np.sum(lspectrum[0][efilter]**2 / lspectrum[1][efilter])
    n = np.sum(efilter)

    d = sumxsqovery * sumoneovery - sumxovery * sumxovery
    r = (sumx * sumoneovery - n * sumxovery) / d
    a = np.exp((n * sumxsqovery / d - sumx * sumxovery / d))
    print('\nFit coefficients I(E) = A*E^r\n')
    print(f'r = {r} A = {a}\n')

    back = spectrum.copy()
    back[1] = a * spectrum[0]**r
    core = spectrum.copy()
    core[1] = spectrum[1] - back[1]
    corefilt = (core[0] >= (eprestart + eprewidth)) & (core[0] <= (epoststart + epostwidth)) & (core[1] > 0)
    core = core[:, corefilt]

    plt.figure()
    plt.plot(spectrum[0], spectrum[1], 'k', linewidth=2)
    plt.title('AFit')
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('Counts')
    plt.plot(core[0], core[1], 'g', linewidth=2)
    plt.plot(back[0], back[1], 'r', linewidth=2)
    plt.legend(['Spectrum', 'Core', 'AFit'])
    plt.show()

    if outcore is None or outback is None:
        outcore = input('filename for core: ')
        outback = input('filename for background: ')
    else:
        print(f'filename for core: {outcore}')
        print(f'filename for background: {outback}')

    np.savetxt(outcore, core.T, fmt='%f')
    np.savetxt(outback, back.T, fmt='%f')


if __name__ == '__main__':
    Afit()