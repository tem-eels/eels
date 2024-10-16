import numpy as np
from scipy.integrate import quad

# Global variables for energy levels and screening factors
IE3 = [73, 99, 135, 164, 200, 245, 294, 347, 402, 455, 513, 575, 641, 710, 779, 855, 931, 1021, 1115, 1217, 1323, 1436, 1550, 1675]
XU = [0.52, 0.42, 0.30, 0.29, 0.22, 0.30, 0.22, 0.16, 0.12, 0.13, 0.13, 0.14, 0.16, 0.18, 0.19, 0.22, 0.14, 0.11, 0.12, 0.12, 0.12, 0.10, 0.10, 0.10]
IE1 = [118, 149, 189, 229, 270, 320, 377, 438, 500, 564, 628, 695, 769, 846, 926, 1008, 1096, 1194, 1142, 1248, 1359, 1476, 1596, 1727]

def gosfunc(E, qa02, z):
    """
    gosfunc calculates (=DF/DE) which IS PER EV AND PER ATOM
    """
    r = 13.606
    zs = z - 0.35 * (8 - 1) - 1.7
    iz = int(z) - 12
    u = XU[iz-1]
    el3 = IE3[iz-1]
    el1 = IE1[iz-1]

    q = qa02 / zs**2
    kh2 = (E / (r * zs**2)) - 0.25
    akh = np.sqrt(abs(kh2))

    if kh2 >= 0:
        d = 1 - np.exp(-2 * np.pi / akh)
        bp = np.arctan(akh / (q - kh2 + 0.25))
        if bp < 0:
            bp += np.pi
        c = np.exp((-2 / akh) * bp)
    else:
        d = 1
        c = np.exp((-1 / akh) * np.log((q + 0.25 - kh2 + akh) / (q + 0.25 - kh2 - akh)))

    if E - el1 <= 0:
        g = (2.25 * q**4 - (0.75 + 3 * kh2) * q**3 + 
             (0.59375 - 0.75 * kh2 - 0.5 * kh2**2) * q**2 +
             (0.11146 + 0.85417 * kh2 + 1.8833 * kh2**2 + kh2**3) * q +
             0.0035807 + kh2 / 21.333 + kh2**2 / 4.5714 + kh2**3 / 2.4 + kh2**4 / 4)
        a = ((q - kh2 + 0.25)**2 + kh2)**5
    else:
        g = (q**3 - (5 / 3 * kh2 + 11 / 12) * q**2 + 
             (kh2**2 / 3 + 1.5 * kh2 + 65 / 48) * q +
             kh2**3 / 3 + 0.75 * kh2**2 + 23 / 48 * kh2 + 5 / 64)
        a = ((q - kh2 + 0.25)**2 + kh2)**4

    rf = ((E + 0.1 - el3) / 1.8 / z / z)**u
    if abs(iz - 11) <= 5 and E - el3 <= 20:
        rf = 1

    return rf * 32 * g * c / a / d * E / r / r / zs**4

def Sigmal3(z=None, delta1=None, e0=None, beta=None):
    """
    SIGMAL3 : CALCULATION OF L-SHELL CROSS-SECTIONS USING A
    MODIFIED HYDROGENIC MODEL WITH RELATIVISTIC KINEMATICS.
    """

    print('\n----------------Sigmal3---------------\n\n')

    if z is None or delta1 is None or e0 is None or beta is None:
        print('Alternate Usage: Sigmal3(Z, Delta(eV), E0 (keV), Beta (mrad))\n')
        z = int(input('Atomic number Z: '))
        delta1 = float(input('Integration window Delta (eV): '))
        e0 = float(input('Incident-electron energy E0 (keV): '))
        beta = float(input('Collection semi-angle Beta (mrad): '))
    else:
        print(f'Atomic number Z: {z}')
        print(f'Integration window Delta(eV): {delta1}')
        print(f'Incident-electron energy E0(keV): {e0}')
        print(f'Collection semi-angle Beta(mrad): {beta}')

    einc = delta1 / 10
    r = 13.606
    iz = int(z) - 12
    el3 = IE3[iz-1]

    e = el3
    b = beta / 1000
    t = 511060 * (1 - 1 / (1 + e0 / 511.06)**2) / 2
    gg = 1 + e0 / 511.06
    p02 = t / r / (1 - 2 * t / 511060)
    f = 0
    s = 0
    sigma = 0

    print('\nE(eV)    ds/dE(barn/eV)  Delta(eV) Sigma(barn^2)     f(0)\n')

    for j in range(1, 41):
        qa021 = e**2 / (4 * t * r) + e**3 / (8 * r * t**2 * gg**3)
        pp2 = p02 - e / r * (gg - e / 1022120)
        qa02m = qa021 + 4 * np.sqrt(p02 * pp2) * (np.sin(b / 2))**2

        dsbyde = 3.5166e8 * (r / t) * (r / e) * quad(lambda x: gosfunc(e, np.exp(x), z), np.log(qa021), np.log(qa02m))[0]
        dfdipl = gosfunc(e, qa021, z)
        delta = e - el3
        if j != 1:
            s = np.log(dsbdep / dsbyde) / np.log(e / (e - einc))
            sginc = (e * dsbyde - (e - einc) * dsbdep) / (1 - s)
            sigma += sginc
            f += (dfdipl + dfprev) * einc / 2
            if delta >= 10:
                print(f'{e:4g} {dsbyde:17.6f} {int(delta):10d} {sigma:13.2f} {f:8.4f}')
        if delta >= delta1:
            if sginc < 0.001 * sigma:
                break
            einc *= 2
        e += einc
        if e > t:
            e -= einc
            break
        dfprev = dfdipl
        dsbdep = dsbyde

if __name__ == "__main__":
    Sigmal3()