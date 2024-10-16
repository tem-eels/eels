import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from ReadData import ReadData

def Kroeger(infile=None, ee=None, thick=None, ang=None):
    """
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    """

    print("\n---------------Kroeger----------------\n\n")

    if infile is None or any(param is None for param in [ee, thick, ang]):
        print("Alternate Usage: Kroeger('infile', E0, t, ang)\n\n")
        infile = input('Input data file (e.g. drude.eps) = ')
        ee = float(input('E0 (keV): '))
        thick = float(input('thickness (nm): '))
        ang = float(input('Collection Angle (mrad): '))
    else:
        print(f'Input data file = {infile}')
        print(f'E0 (keV): {ee}')
        print(f'thickness (nm): {thick}')
        print(f'Collection Angle (mrad): {ang}')

    # Read data from file
    numCols = 3
    inData = ReadData(infile, numCols) 

    # Determine data size
    dsize = inData.shape[0] - 1
    hdsize = (dsize + 1) / 2

    # Extract energy loss and dielectric function data and take its conjugate
    edata = np.real(inData[:, 0])

    # if 3 column data combine column 2 and 3 to form complex number
    # if 2 column data assume column 2 is complex number
    if numCols == 3:
        epsdata = np.vectorize(complex)(inData[:, 1], -inData[:, 2])  # conjugate
    elif numCols == 2:
        epsdata = np.conj(inData[:, 1])
    else:
        raise ValueError('eps data has too many columns')

    # Adjust input to SI units
    thick = thick * 1e-9  # thickness in meters
    ang = ang * 1e-3  # angular range in radians

    # Generate sample scattering angles
    # The reason we break down into two arrays is to mimic how colon operator works in MATLAB
    # http://web.archive.org/web/20120213165003/http://www.mathworks.com/support/solutions/en/data/1-4FLI96/index.html
    increment = ang / (hdsize - 0.5)
    adata_left = np.arange(-ang, 0, increment)
    adata_right = np.arange(ang, 0-increment/2, -increment)
    if adata_left[-1] == adata_right[-1]:
        adata_right = adata_right[:-1]
    #reverse right array
    adata_right = adata_right[::-1]
    adata = np.concatenate((adata_left, adata_right))
    # Shift Energy data
    edata += 1e-8

    # Get solution of Kroeger Equation to use in plotData
    P, Pvol = KroegerCore(edata, adata, epsdata, ee, thick)
    np.savetxt('Kroeger_P.txt', P, delimiter='\n', fmt='%.18f')
    np.savetxt('Kroeger_Pvol.txt', Pvol, delimiter='\n', fmt='%.18f')
    np.savetxt('Kroeger_edata.txt', edata, delimiter='\n', fmt='%.18f')
    np.savetxt('Kroeger_adata.txt', adata, delimiter='\n', fmt='%.18f')

    # Plot dielectric data used in the calculation
    ploteV(edata, np.vstack((np.real(epsdata), -np.imag(epsdata))).T,
           'Kroeger input data', '', False, legend=['eps1', 'eps2'])
    # print(adata[adata >= 0])
    # print(edata[edata >= 1])
    # Create surface plots using ploteVTheta and angular dependence at fixed eV
    ploteVTheta(edata, adata, Pvol, 'Kroeger(bulk only)', 'd^2P/(dΩ dE) [1/(eV sr)]')
    ploteVTheta(edata, adata, P, 'Kroeger (all terms)', 'd^2P/(dΩ dE) [1/(eV sr)]')
    plotTheta(adata[adata >= 0], P[adata >= 0, np.searchsorted(edata, 1)], '1eV total', 'P=d^2P/dΩ.dE')
    plotTheta(adata[adata >= 0], P[adata >= 0, np.searchsorted(edata, 5)], '5eV total', 'P=d^2P/dΩ.dE')
    plotTheta(adata[adata >= 0], P[adata >= 0, np.searchsorted(edata, 13)], '13eV total', 'P=d^2P/dΩ.dE')
    ploteV(edata[edata >= 1], P[np.where(adata >= 0)[0][0], edata >= 1], 'All terms, theta=0', 'P=d^2P/dΩ.dE', False)
    np.savetxt('Kroeger_allterm.txt', P[np.where(adata >= 0)[0][0], edata >= 1], delimiter='\n', fmt='%.18f')


    # Integrate over detection area
    intP = IntegrateTheta(0, ang, edata, epsdata, ee, thick)
    np.savetxt('Kroeger_intP.txt', intP, delimiter='\n', fmt='%.18f')
    ploteV(edata, intP, 'Angle-integrated probability', 'dP/dE [1/eV]', False)

    # Integrate over energy loss and print result
    epc = edata[1] - edata[0]
    intintP = np.trapz(intP, edata) / epc
    print(f'\nP(vol+surf) = {intintP}\n')

def KroegerCore(edata, adata, epsdata, ee, thick):
    """
    Core Kroeger function as defined in E&B Eq. 4
    """
    
    # Constants
    mel = 9.109e-31  # Electron mass in kg
    h = 6.626068e-34  # Planck's constant
    hbar = h / (2 * np.pi)
    c = 2.99792458e8  # Speed of light in m/s
    bohr = 5.2918e-11  # Bohr radius in meters
    e = 1.60217646e-19  # Electron charge in Coulomb

    # Calculate fixed terms of the equation
    va = 1 - (511 / (511 + ee)) ** 2  # ee is incident energy in keV
    v = c * np.sqrt(va)
    beta = v / c
    gamma = 1 / np.sqrt(1 - beta ** 2)
    momentum = mel * v * gamma

    # Meshgrid for E, Theta, and eps
    E, Theta = np.meshgrid(edata, adata)
    eps, _ = np.meshgrid(epsdata, adata)

    Theta2 = Theta ** 2 + 1e-15
    ThetaE = E * e / momentum / v
    ThetaE2 = ThetaE ** 2

    lambda2 = Theta2 - eps * ThetaE2 * beta ** 2
    lambda_ = np.emath.sqrt(lambda2)
    if not np.all(np.real(lambda_) >= 0):
        raise ValueError('Negative lambda')

    phi2 = lambda2 + ThetaE2
    lambda02 = Theta2 - ThetaE2 * beta ** 2
    lambda0 = np.emath.sqrt(lambda02)
    if not np.all(np.real(lambda0) >= 0):
        raise ValueError('Negative lambda0')

    de = thick * E * e / 2 / hbar / v
    xy = lambda_ * de / ThetaE
    lplus = lambda0 * eps + lambda_ * np.tanh(xy)
    lminus = lambda0 * eps + lambda_ / np.tanh(xy)

    mue2 = 1 - eps * beta ** 2
    phi20 = lambda02 + ThetaE2
    phi201 = Theta2 + ThetaE2 * (1 - (eps + 1) * beta ** 2)

    Pcoef = e / (bohr * np.pi ** 2 * mel * v ** 2)
    Pv = thick * mue2 / eps / phi2

    Ps1 = 2 * Theta2 * (eps - 1) ** 2 / phi20 ** 2 / phi2 ** 2
    Ps2 = hbar / momentum

    A1 = phi201 ** 2 / eps
    A2 = np.sin(de) ** 2 / lplus + np.cos(de) ** 2 / lminus
    A = A1 * A2

    B1 = beta ** 2 * lambda0 * ThetaE * phi201
    B2 = (1 / lplus - 1 / lminus) * np.sin(2 * de)
    B = B1 * B2

    C1 = -beta ** 4 * lambda0 * lambda_ * ThetaE2
    C2 = np.cos(de) ** 2 * np.tanh(xy) / lplus
    C3 = np.sin(de) ** 2 / np.tanh(xy) / lminus
    C = C1 * (C2 + C3)

    Ps3 = A + B + C
    Ps = Ps1 * Ps2 * Ps3

    P = np.real(Pcoef) * np.imag(Pv - Ps)
    Pvol = np.real(Pcoef) * np.imag(Pv)

    return P, Pvol

def IntegrateTheta(startang, endang, edata, epsdata, ee, thick):
    """
    Integrate over scattering angle.
    startang - starting angle for integration in rads
    endang - ending angle for integration in rads
    edata - energy loss channels in eV
    epsdata - conjugate dielectric data (data must be same size as edata)
    ee - electron energy in keV
    thick - specimen thickness in meters
    """
    intDatal = np.zeros(len(edata))

    for i in range(len(edata)):
        PSin = lambda inTheta: KroegerCore([edata[i]], [inTheta], [epsdata[i]], ee, thick)[0] * np.sin(inTheta)
        intDatal[i] = 2 * np.pi * quad(PSin, startang, endang, limit=10000, epsabs=1.0e-6)[0]

    return intDatal

def ploteVTheta(xscale, yscale, M, Mtitle, Mzlabel, logbool=True):
    """
    Creates a surface plot of a function M(x, y) where units for x are in eV and y are in rad
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(xscale, yscale)

    if logbool:
        M = np.maximum(M, 1e-15)  # Avoid taking log of zero or negative numbers
        Z = np.log10(M)  # Apply log10 scaling
        surf = ax.plot_surface(X, Y, Z, cmap='jet')
        colorbar = fig.colorbar(surf, ax=ax)
        colorbar.set_label(Mzlabel + ' (log scale)')
    else:
        surf = ax.plot_surface(X, Y, np.real(M), cmap='jet')
        colorbar = fig.colorbar(surf, ax=ax)
        colorbar.set_label(Mzlabel)

    ax.set_xlabel('Energy Loss [eV]')
    ax.set_ylabel('Scattering Angle [rad]')
    ax.set_zlabel(Mzlabel)
    plt.title(Mtitle, fontsize=12)
    plt.show()

def plotTheta(xscale, M, Mtitle, Mylabel, logbool=True):
    """
    Creates a line plot for a function M(x) where units of x are in rad
    """
    plt.figure()
    if logbool:
        M = np.maximum(M, 1e-15)
        plt.plot(xscale, np.real(M))
        plt.yscale('log')
    else:
        plt.plot(xscale, np.real(M))

    plt.title(Mtitle, fontsize=12)
    plt.xlabel('Scattering Angle [rad]')
    plt.ylabel(Mylabel)
    plt.show()

def ploteV(xscale, M, Mtitle, Mylabel, logbool=True, legend=None):
    """
    Creates a line plot for a function M(x) where units of x are in eV
    """
    plt.figure()
    if logbool:
        M = np.where(M > 0, M, np.nan)
        plt.plot(xscale, np.real(M))
        plt.yscale('log')
    else:
        plt.plot(xscale, np.real(M))

    plt.title(Mtitle, fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel(Mylabel)
    if legend is not None:
        plt.legend(legend)
    plt.show()

if __name__ == "__main__":
    Kroeger()