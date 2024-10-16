import numpy as np
from scipy.integrate import quad

def gosfunc(E, qa02, z):
    """
    gosfunc calculates (=DF/DE) which IS PER EV AND PER ATOM
    """
    r = 13.606
    zs = 1.0
    rnk = 1
    if z != 1:
        zs = z - 0.50
        rnk = 2
    q = qa02 / zs**2
    kh2 = E / r / zs**2 - 1
    akh = np.sqrt(abs(kh2))
    if akh <= 0.1:
        akh = 0.1
    if kh2 >= 0.0:
        d = 1 - np.exp(-2.0 * np.pi / akh)
        bp = np.arctan(2.0 * akh / (q - kh2 + 1))
        if bp < 0:
            bp = bp + np.pi
        c = np.exp((-2.0 / akh) * bp)
    else:
        d = 1
        y = (-1.0 / akh * np.log((q + 1 - kh2 + 2 * akh) / (q + 1 - kh2 - 2 * akh)))
        c = np.exp(y)
    a = ((q - kh2 + 1)**2 + 4.0 * kh2)**3
    return 128 * rnk * E / r / zs**4 * c / d * (q + kh2 / 3 + 1 / 3) / a / r

def Sigmak3(z=None, ek=None, delta1=None, e0=None, beta=None):
    """
    SIGMAK3 : CALCULATION OF K-SHELL IONIZATION CROSS SECTIONS
    USING RELATIVISTIC KINEMATICS AND A HYDROGENIC MODEL WITH
    INNER-SHELL SCREENING CONSTANT OF 0.5.
    """

    print('\n---------------Sigmak3----------------\n\n')

    if z is None or ek is None or delta1 is None or e0 is None or beta is None:
        print('Alternate Usage: Sigmak3(Z, EK, Delta, E0, Beta)\n')
        z = int(input('Atomic number Z: '))
        ek = float(input('K-edge threshold energy EK (eV): '))
        delta1 = float(input('Integration window Delta (eV): '))
        e0 = float(input('Incident-electron energy E0 (keV): '))
        beta = float(input('Collection semi-angle Beta (mrad): '))
    else:
        print(f'Atomic number Z: {z}')
        print(f'K-edge threshold energy EK (eV): {ek}')
        print(f'Integration window Delta (eV): {delta1}')
        print(f'Incident-electron energy E0 (keV): {e0}')
        print(f'Collection semi-angle Beta (mrad): {beta}')

    einc = delta1 / 10
    r = 13.606
    e = ek
    b = beta / 1000
    t = 511060 * (1 - 1 / (1 + e0 / 511.06)**2) / 2
    gg = 1 + e0 / 511.06
    p02 = t / r / (1 - 2 * t / 511060)
    f = 0
    s = 0
    sigma = 0
    dsbdep = 0
    dfprev = 0

    print('\nE(eV)    ds/dE(barn/eV)  Delta(eV)   Sigma(barn)     f(0)\n')
    
    for j in range(1, 31):
        qa021 = e**2 / (4 * r * t) + e**3 / (8 * r * t**2 * gg**3)
        pp2 = p02 - e / r * (gg - e / 1022120)
        qa02m = qa021 + 4 * np.sqrt(p02 * pp2) * (np.sin(b / 2))**2

        dsbyde = 3.5166e8 * (r / t) * (r / e) * quad(lambda x: gosfunc(e, np.exp(x), z), np.log(qa021), np.log(qa02m))[0]
        dfdipl = gosfunc(e, qa021, z)
        delta = e - ek
        if j != 1:
            s = np.log(dsbdep / dsbyde) / np.log(e / (e - einc))
            sginc = (e * dsbyde - (e - einc) * dsbdep) / (1 - s)
            sigma += sginc
            f += (dfdipl + dfprev) / 2 * einc
        
        print(f'{e:4g} {dsbyde:17.6f} {int(delta):10d} {sigma:13.2f} {f:8.4f}')
        
        if einc == 0:
            return
        if delta >= delta1:
            if sginc < 0.001 * sigma:
                break
            einc *= 2
        e += einc
        if e > t:
            break
        dfprev = dfdipl
        dsbdep = dsbyde

if __name__ == "__main__":
    Sigmak3()