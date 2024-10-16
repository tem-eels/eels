import numpy as np

def SigADF(Z=None, A=None, qn=None, qx=None, E0=None):
    """
    Calculates high-angle cross sections for high-angle elastic scattering,
    as utilized in an annular dark-field (ADF) detector.

    Based on an analytical formula (Banhart, 1999; Eq.5) using the
    McKinley-Feshbach (1948) approximation, valid for Z < 28.
    
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition,
    with correction made (8 Sept. 2011) to Eq.(3.12f) and to Mott cross section.
    """
    
    print("\n------------SigADF-------------\n\n")
    print("SigADF: HAADF elastic cross sections\n")
    
    if Z is None or A is None or qn is None or qx is None or E0 is None:
        print("Alternate Usage: SigADF(Z, A, minAng, maxAng, E0)\n\n")
        
        Z = float(input("Atomic number Z: "))
        A = float(input("Atomic weight A: "))
        qn = float(input("Minimum scattering angle (mrad): "))
        qx = float(input("Maximum scattering angle (mrad): "))
        E0 = float(input("Incident-electron energy E0 (keV): "))
    else:
        print(f"Atomic number Z: {Z}")
        print(f"Atomic weight A: {A}")
        print(f"Minimum scattering angle (mrad): {qn}")
        print(f"Maximum scattering angle (mrad): {qx}")
        print(f"Incident-electron energy E0 (keV): {E0}")
    
    Al = Z / 137  # fine-structure constant
    a0 = 5.29e-11  # in meters
    R = 13.606  # Bohr radius in meters, Rydberg energy in eV
    t = E0 * (1 + E0 / 1022) / (1 + E0 / 511) ** 2  # m0v^2/2
    b = np.sqrt(2 * t / 511)  # v/c
    k0 = 2590e9 * (1 + E0 / 511) * b  # in meters^-1
    q0 = 1000 * Z ** 0.3333 / k0 / a0  # Lenz screening angle in mrad
    
    print(f"\nLenz screening angle = {q0:.6g} mrad\n")
    if q0 > qn:
        print("WARNING: minimum angle < Lenz screening angle!")
    
    newrf = (1 - b ** 2) / b ** 4  # relativistic factor
    smin = np.sin(qn/2000) ** 2
    smax = np.sin(qx/2000) ** 2
    x = 1 / smin - 1 / smax  # spherical potential
    sdc = 0.2494 * Z ** 2 * newrf * x  # Rutherford cross section
    print(f"Rutherford cross section = {sdc:g} barn")
    
    coef = 4 * Z ** 2 * R ** 2 / (511000) ** 2
    sqb = 1 + 2 * np.pi * Al * b + (b ** 2 + np.pi * Al * b) * np.log(x)
    brace = 1 + 2 * np.pi * Al * b / np.sqrt(x) - sqb / x
    sdmf = 9.999999e27 * coef * x * np.pi * a0 ** 2 * (1 - b ** 2) / b ** 4 * brace
    print(f"McKinley-Feshbach-Mott cross section = {sdmf:g} barn")

if __name__ == "__main__":
    SigADF()