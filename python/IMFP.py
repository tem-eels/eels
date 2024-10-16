import numpy as np

def IMFP(Z=None, A=None, frac=None, alpha=None, beta=None, rho=None, E0=None):
    """
    Generate inelastic mean free path based on parameterizations of
    Iakoubovskii (2008), with and without correction of thetaE expression
    Jin and Li (2006), and Malis et al. (1988).
    """
    print('\n--------------IMFP-----------------\n\n')

    if Z is None or A is None or frac is None:
        Z, A, frac = [], [], []
        sumFrac = 0
        idx = 0
        while sumFrac < 1:
            idx += 1
            print(f'Enter values for element {idx}')
            z = int(input('  Enter atomic number Z (Enter 0 if not known): '))
            a = float(input('  Enter atomic weight A (Enter 0 if not known): '))
            f = float(input(f'  Enter compound fraction ({1-sumFrac:g} remaining): '))
            Z.append(z)
            A.append(a)
            frac.append(f)
            sumFrac += f
    
    Z = np.array(Z)
    A = np.array(A)
    frac = np.array(frac)
    
    sumFrac = np.sum(frac)
    if sumFrac > 1:
        print(f'WARNING: Sum of element fractions exceeds 1, reducing fraction of element {len(Z)} by {sumFrac - 1:g}')
        frac[-1] -= (sumFrac - 1)
    elif sumFrac < 1:
        raise ValueError('Sum of element fractions below 1')

    print('\nCompound Summary')
    for i in range(len(Z)):
        print(f'Element: {i+1} Z: {Z[i]} A: {A[i]} Fraction: {frac[i]:g}')
    print('\n')

    if alpha is None or beta is None or rho is None or E0 is None:
        alpha = float(input('Enter incident-convergence semi-angle alpha (mrad): '))
        beta = float(input('Enter EELS collection semi-angle beta (mrad): '))
        rho = float(input('Enter specimen density rho (g/cm3) (Enter 0 if not known): '))
        E0 = float(input('Enter incident energy E0 (keV): '))
    else:
        print(f'Incident-convergence semi-angle alpha (mrad): {alpha}')
        print(f'EELS collection semi-angle beta (mrad): {beta}')
        print(f'Specimen density rho (g/cm3): {rho}')
        print(f'Incident energy E0 (keV): {E0}')
    
    F = (1 + E0 / 1022) / (1 + E0 / 511) ** 2
    Fg = (1 + E0 / 1022) / (1 + E0 / 511)
    TGT = 2 * Fg * E0
    print(f'2.gamma.T = {TGT:g}\n')

    a2 = alpha ** 2
    b2 = beta ** 2

    if Z.size > 0:
        Zef = np.sum(frac * Z ** 1.3) / np.sum(frac * Z ** 0.3)
    else:
        Zef = 0
    print(f'Effective atomic number Z: {Zef:g}')
    
    if A.size > 0:
        Aef = np.sum(frac * A ** 1.3) / np.sum(frac * A ** 0.3)
    else:
        Aef = 0
    print(f'Effective atomic weight A: {Aef:g}\n')

    if rho > 0:
        qE2 = (5.5 * rho ** 0.3 / (F * E0)) ** 2
        qc2 = 400
        coeff = (11 * rho ** 0.3) / (200 * F * E0)
        num = (a2 + b2 + 2 * qE2 + abs(a2 - b2)) * qc2
        den = (a2 + b2 + 2 * qc2 + abs(a2 - b2)) * qE2
        LiI = 1 / (coeff * np.log(num / den))
        qE2g = (5.5 * rho ** 0.3 / (Fg * E0)) ** 2
        num2 = (a2 + b2 + 2 * qE2g + abs(a2 - b2)) * qc2
        den2 = (a2 + b2 + 2 * qc2 + abs(a2 - b2)) * qE2g
        LiI2 = 1 / (coeff * np.log(num2 / den2))
        print(f'IMFP (Iakoubovskii, 2008) = {LiI:g} nm')
        print(f'IMFP (Iakoubovskii revised) = {LiI2:g} nm\n')
    else:
        print('IMFP (Iakoubovskii) not calculated since density = 0\n')

    if Zef > 0:
        e = 13.5 * Zef / 2
        tgt = E0 * (1 + E0 / 1022) / (1 + E0 / 511)
        thetae = (e + 1e-6) / tgt
        a2_rad = alpha ** 2 * 1e-6 + 1e-10
        b2_rad = beta ** 2 * 1e-6
        t2 = thetae ** 2 * 1e-6

        eta1 = np.sqrt((a2_rad + b2_rad + t2) ** 2 - 4 * a2_rad * b2_rad) - a2_rad - b2_rad - t2
        eta2 = 2 * b2_rad * np.log(0.5 / t2 * (np.sqrt((a2_rad + t2 - b2_rad) ** 2 + 4 * b2_rad * t2) + a2_rad + t2 - b2_rad))
        eta3 = 2 * a2_rad * np.log(0.5 / t2 * (np.sqrt((b2_rad + t2 - a2_rad) ** 2 + 4 * a2_rad * t2) + b2_rad + t2 - a2_rad))
        eta = (eta1 + eta2 + eta3) / a2_rad / np.log(4 / t2)
        f1 = (eta1 + eta2 + eta3) / (2 * a2_rad * np.log(1 + b2_rad / t2))
        f2 = f1
        if alpha / beta > 1:
            f2 = f1 * a2_rad / b2_rad
        
        bstar = thetae * np.sqrt(np.exp(f2 * np.log(1 + b2_rad / t2)) - 1)
        print(f'F1 = {f1:g}')
        print(f'F2 = {f2:g}')
        print(f'beta* = {bstar:g} mrad\n')

        Em = 7.6 * Zef ** 0.36
        LiM = 106 * F * E0 / Em / np.log(2 * bstar * E0 / Em)
        print(f'IMFP (Malis et al.) = {LiM:g} nm')

        if rho > 0 and Aef > 0:
            Em = 42.5 * Zef ** 0.47 * rho / Aef
            LiJL = 106 * F * E0 / Em / np.log(2 * bstar * E0 / Em)
            print(f'IMFP (Jin & Li) = {LiJL:g} nm\n')
        else:
            print('IMFP (Jin & Li) not calculated since A = 0 OR density = 0\n')
    else:
        print('IMFP (Malis et al.) & IMFP (Jin & Li) not calculated since Z = 0\n')

if __name__ == "__main__":
    IMFP()