import numpy as np

def Sigdis(Z=None, A=None, Ed=None, E0=None):
    """
    Sigdis: Calculates cross sections for bulk atomic displacement or 
    surface sputtering for both spherical and planar escape potentials.
    Based on an analytical formula (Banhart, 1999; Eq.5) that uses the
    McKinley-Feshbach (1948) approximation, valid for Z < 28.
    For Z > 28, the Rutherford value should be used as a better approximation.
    
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    """
    
    print("\n-------------Sigdis------------\n\n")
    print("Sigdis: Atomic-displacement cross sections:\n")
    
    if Z is None or A is None or Ed is None or E0 is None:
        print("Alternate Usage: Sigdis(Z, A, Ed, E0)\n\n")
        
        Z = float(input("Atomic number Z: "))
        A = float(input("Atomic weight A: "))
        Ed = float(input("Surface or bulk displacement energy Ed (eV): "))
        E0 = float(input("Incident-electron energy E0 (keV): "))
    else:
        print(f"Atomic number Z: {Z}")
        print(f"Atomic weight A: {A}")
        print(f"Surface or bulk displacement energy Ed (eV): {Ed}")
        print(f"Incident-electron energy E0 (keV): {E0}")
    
    Al = Z / 137
    A0 = 5.29e-11
    R = 13.606  # Bohr radius in meters, Rydberg energy in eV
    t = E0 * (1 + E0 / 1022) / (1 + E0 / 511) ** 2
    b = np.sqrt(2 * t / 511)
    newrf = (1 - b**2) / b**4
    Emax = (E0 / A) * (E0 + 1022) / 465.7
    E0min = 511 * (np.sqrt(1 + A * Ed / 561) - 1)
    
    print(f"Emax(eV) = {Emax:.6g} eV, threshold = {E0min:.6g} keV")
    
    coef = 4 * Z**2 * R**2 / (511000)**2
    Emin = np.sqrt(Ed * Emax)  # for planar potential, otherwise Emin = Ed
    print(f"Emin(planar potential) = {Emin:.6g} eV")
    
    x = Emax / Ed  # spherical potential
    xp = Emax / np.sqrt(Ed * Emax)  # planar potential
    
    sdc = 0.2494 * Z**2 * newrf * (x - 1)
    pdc = 0.2494 * Z**2 * newrf * (xp - 1)
    
    print(f"Rutherford value (spherical escape potential) = {sdc:.6g} barn")
    print(f"Rutherford value (planar escape potential) = {pdc:.6g} barn")
    
    sqb = 1 + 2 * np.pi * Al * b + (b**2 + np.pi * Al * b) * np.log(x)
    brace = 1 + 2 * np.pi * Al * b / np.sqrt(x) - sqb / x
    
    sdmf = 9.999999e27 * coef * x * np.pi * A0**2 * (1 - b**2) / b**4 * brace
    print(f"McKinley-Feshbach-Mott (spherical potential) = {sdmf:.6g} barn")
    
    psqb = 1 + 2 * np.pi * Al * b + (b**2 + np.pi * Al * b) * np.log(xp)
    pbrace = 1 + 2 * np.pi * Al * b / np.sqrt(xp) - psqb / xp
    
    pdmf = 9.999999e27 * coef * xp * np.pi * A0**2 * (1 - b**2) / b**4 * pbrace
    print(f"McKinley-Feshbach-Mott (planar potential) = {pdmf:.6g} barn")

if __name__ == "__main__":
    Sigdis()