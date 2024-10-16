import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from Kroeger import KroegerCore, IntegrateTheta
from ReadData import ReadData

def KroegerEBplots(infile=None, E0=None, el=None, beta=None):
    """
    This function takes a set of energy channels and their corresponding
    dielectric values and creates a set of plots similar to ones found in E&B
    Fig 3 & 4a.

    Parameters:
    - infile: Input data file (string)
    - E0: Beam energy in keV (float)
    - el: Energy loss for angular distribution in eV (float)
    - beta: Angular range in mrad (float)
    """

    print('\n---------------KroegerEBplots----------------\n')

    if infile is None or E0 is None or el is None or beta is None:
        infile = input('Input data file (e.g. KroegerEBplots_Si.dat) = ')
        E0 = float(input('E0 (keV): '))
        el = float(input('Energy loss for angular distribution (eV): '))
        beta = float(input('Angular Range (mrad): '))
    else:
        print(f'Input data file = {infile}')
        print(f'E0 (keV): {E0}')
        print(f'Energy loss for angular distribution (eV): {el}')
        print(f'Angular Range (mrad): {beta}')

    numCols = 3
    inData = ReadData(infile, numCols)

    # Extract energy loss and dielectric function data and take its conjugate
    edata = inData[:, 0]
    edata = edata + 1e-8  # Avoid division by zero

    # Combine columns to form complex dielectric function data
    if numCols == 3:
        epsdata = np.complex128(inData[:, 1] - 1j * inData[:, 2])  # conjugate
    elif numCols == 2:
        epsdata = np.conj(inData[:, 1])
    else:
        raise ValueError('eps data has too many columns')

    # Get height of 1 nm plot to normalize with (Fig. 4a)
    temp = IntegrateTheta(0, beta * 1e-3, edata, epsdata, E0, 10e-10)

    max10 = np.max(temp)

    TSthick = np.array([100000, 10000, 5000, 2500, 1000, 500, 250, 100, 50, 10])
    evFix = np.where(edata >= el)[0][0]  # Set fixed eV to ~ el

    TSintP = np.zeros((len(TSthick), len(edata)))
    angles = np.arange(100, 0, -100/512)
    # append 0
    angles = np.append(angles, 0)
    angles_pre = angles[::-1]
    angles = angles_pre * 1e-6

    TSPeVFixed = np.zeros((len(TSthick), len(angles)))
    for i, thickness in enumerate(TSthick):
        print(f'Currently processing {thickness/10:.0f} nm')
        # Integrate for specific thickness (Fig 4a)
        TSintP[i, :] = IntegrateTheta(1e-8, beta*1e-3, edata, epsdata, E0, thickness*1e-10)

        # Normalize height of plot using max10 and shift data upward (Fig 4a)
        maxy = np.max(TSintP[i, :])
        TSintP[i, :] = TSintP[i, :] * max10 / maxy + max10*(10-(i+1))/2

        # Kroeger for specific energy (edata[evFix]) and thickness (Fig. 3)
        P, _ = KroegerCore(edata[evFix], angles, epsdata[evFix], E0, thickness * 1e-10)
        P = P.flatten()
        TSPeVFixed[i, :] = P * 10 ** ((10 - (i+1)) / 2.0)  # Apply y shift as in Fig 3
    TSintP = np.array(TSintP)
    TSPeVFixed = np.array(TSPeVFixed)

    # Plot for Fig 3
    plt.figure()
    TSadata = np.tile(angles_pre, (len(TSthick), 1))
    plt.plot(TSadata.T * 1e-3, np.log10(TSPeVFixed.T))
    plt.legend(['10000nm -4.5', '1000nm -4.0', '5000nm -3.5', '2500nm -3.0', '1000nm -2.5',
                '500nm -2.0', '250nm -1.5', '100nm -1.0', '50nm -0.5', '10nm'])
    plt.title(f'{E0}keV, {edata[evFix]} eV', fontsize=12)
    plt.xlabel('Scattering Angle [mrad]')
    plt.ylabel('log10(P)[1/(eV srad)]')
    plt.yticks(np.arange(0, 14, 1))

    # Plot for Fig 4a
    plt.figure()
    TSedata = np.tile(edata, (len(TSthick), 1))
    plt.plot(TSedata.T, TSintP.T)
    plt.legend(['10000nm', '1000nm', '5000nm', '2500nm', '1000nm',
                '500nm', '250nm', '100nm', '50nm', '10nm'])
    plt.title(f'{E0}keV, {beta} mrads', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel("'Normalized dP/dE [1/eV]'")
    plt.gca().set_yticklabels([])  # Remove y-axis labels

    plt.show()


if __name__ == '__main__':
    KroegerEBplots()