import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def EdgeGen(Ek=None, Emin=None, Emax=None, Ep=None, Epc=None, r=None, TOL=None, JR=None):
    """
    This program generates an idealized hydrogenic ionization-edge profile
    of the form AE^-r but with plural (plasmon+core) scattering added by 
    convolution with a Poisson series of delta-function peaks, to give 
    a series of steps with the correct relative heights; see Eq.(3.117) 
    and Fig. 3.33.
    Background is included, with the same exponent r.
    The order n of plasmon scattering is limited by the requirement that
    AE^-r power law holds only for E > 30 eV
    
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    """
    
    print("\n---------------EdgeGen----------------\n\n")

    if None in [Ek, Emin, Emax, Ep, Epc, r, TOL, JR]:
        print("Alternate Usage: EdgeGen(Ek, Emin, Emax, Ep, Epc, r, TOL, JR)\n\n")
        
        Ek = float(input("Edge energy (eV) = "))
        Emin = float(input("First coreloss channel (eV) = "))
        Emax = float(input("Last coreloss channel (eV) = "))
        Ep = float(input("Plasmon energy (eV) = "))
        Epc = float(input("eV/channel = "))
        r = float(input("Parameter r (e.g. 4) = "))
        TOL = float(input("thickness/inelastic-MFP = "))
        JR = float(input("Jump Ratio = "))
    else:
        print(f"Edge energy (eV) = {Ek}")
        print(f"First coreloss channel (eV) = {Emin}")
        print(f"Last coreloss channel (eV) = {Emax}")
        print(f"Plasmon energy (eV) = {Ep}")
        print(f"eV/channel = {Epc}")
        print(f"Parameter r (e.g. 4) = {r}")
        print(f"thickness/inelastic-MFP = {TOL}")
        print(f"Jump Ratio = {JR}")

    A = Ek ** r * np.exp(TOL)

    Ecore = np.arange(Emin, Emax + Epc, Epc)
    Elow = Ecore - Emin + 1e-5
    core = np.zeros_like(Ecore)
    low = np.zeros_like(Elow)

    # Initialize background
    B = 0

    # Calculate the maximum number of iterations using Emin
    n = int(np.floor((Emin - 30) / Ep))
    print(f"\nCalculated order n = {n}\n")

    for i in range(n):
        J = A * ((Ecore - i * Ep) ** -r) * (TOL ** i) * np.exp(-TOL) / factorial(i)
        maskcore = (Ecore - i * Ep) < Ek
        J[maskcore] = 0
        B += A * ((Ecore - i * Ep) ** -r) * (TOL ** i) * np.exp(-TOL) / factorial(i)
        core += J
        
        masklow = Elow >= Ep * i - Emin + Ek
        if np.any(masklow):
            ploc = np.where(masklow)[0][0]
            low[ploc] += TOL ** i / factorial(i)

    # Target Jump Background Intensity
    TJBI = A * Ek ** -r * np.exp(-TOL) / JR

    # Actual Jump Background Intensity
    CJBI = B[np.where(Ecore >= Ek)[0][0]]

    # Adjusted Background
    B *= TJBI / CJBI

    # Add background to Edge
    core += B

    plt.figure()
    plt.plot(Elow, low)
    plt.title('Low-loss spectrum', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('probability')
    
    plt.figure()
    plt.plot(Ecore, core)
    plt.title('Core-loss + background intensity', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('core-loss intensity')
    plt.show()

    np.savetxt('EdgeGen.cor', np.column_stack((Ecore, core)), fmt='%g %g')
    np.savetxt('EdgeGen.low', np.column_stack((Elow, low)), fmt='%g %g')

if __name__ == "__main__":
    EdgeGen()