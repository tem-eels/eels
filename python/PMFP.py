import numpy as np

def PMFP(E0=None, Ep=None, alpha=None, beta=None):
    """
    Calculate plasmon mean free paths using a free-electron formula.
    """
    if E0 is None or Ep is None or alpha is None or beta is None:
        E0 = float(input('PMFP: Incident-electron energy E0 (keV): '))
        Ep = float(input('Plasmon energy of mean energy loss (eV): '))
        alpha = float(input('Convergence semiangle (mrad) [can be 0]: '))
        beta  = float(input('Collection semiangle (mrad): '))
    else:
        print(f'PMFP: Incident-electron energy E0 (keV): {E0}')
        print(f'Plasmon energy of mean energy loss (eV): {Ep}')
        print(f'Convergence semiangle (mrad) [can be 0]: {alpha}')
        print(f'Collection semiangle (mrad): {beta}')
    
    F = (1 + E0 / 1022) / (1 + E0 / 511) ** 2
    Fg = (1 + E0 / 1022) / (1 + E0 / 511)
    T = E0 * F  # keV
    tgt = 2 * Fg * E0
    a0 = 0.0529  # nm

    thetae = (Ep + 1e-6) / tgt  # in mrad, avoid NaN for e=0
    a2 = alpha**2 * 1e-6 + 1e-10  # radians^2, avoiding inf for alpha=0
    b2 = beta**2 * 1e-6  # radians^2
    t2 = thetae**2 * 1e-6  # radians^2

    eta1 = np.sqrt((a2 + b2 + t2)**2 - 4 * a2 * b2) - a2 - b2 - t2
    eta2 = 2 * b2 * np.log(0.5 / t2 * (np.sqrt((a2 + t2 - b2)**2 + 4 * b2 * t2) + a2 + t2 - b2))
    eta3 = 2 * a2 * np.log(0.5 / t2 * (np.sqrt((b2 + t2 - a2)**2 + 4 * a2 * t2) + b2 + t2 - a2))
    eta = (eta1 + eta2 + eta3) / a2 / np.log(4 / t2)
    f1 = (eta1 + eta2 + eta3) / (2 * a2 * np.log(1 + b2 / t2))
    f2 = f1
    if alpha / beta > 1:
        f2 = f1 * a2 / b2
    
    bstar = thetae * np.sqrt(np.exp(f2 * np.log(1 + b2 / t2)) - 1)  # mrad
    print(f'effective semiangle beta* = {bstar:g} mrad')

    thetabr = 1000 * (Ep / E0 / 1000) ** 0.5
    print(f'Bethe-ridge angle(mrad) = {thetabr:g} nm')

    if bstar < thetabr:
        pmfp = 4000 * a0 * T / Ep / np.log(1 + bstar**2 / thetae**2)
        imfp = 106 * F * E0 / Ep / np.log(2 * bstar * E0 / Ep)
        print(f'Free-electron MFP(nm) = {pmfp:g} nm')
        print(f'Using Eq.(5.2), MFP(nm) = {imfp:g} nm')
    else:
        print('Dipole range is exceeded')
        tmfp = 4000 * a0 * T / Ep / np.log(1 + thetabr**2 / thetae**2)
        print(f'total-inelastic MFP(nm) = {tmfp:g} nm')
    print('\n-------------------------------\n\n')

if __name__ == "__main__":
    PMFP()
