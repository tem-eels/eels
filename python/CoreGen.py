import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def CoreGen(Ek=None, Emin=None, Emax=None, Ep=None, Epc=None, r=None, TOL=None):
    """
    This program generates an idealized hydrogenic ionization-edge profile
    of the form AE^-r but with plural (plasmon+core) scattering added by 
    convolution with a Poisson series of delta-function peaks, to give 
    a series of steps with the correct relative heights.
    It can be used as input to test the Frat.m deconvolution program.
    
    See Eq.(3.117) and Fig. 3.33
    in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    """
    
    print("\n----------------CoreGen---------------\n\n")
    
    # Get input data if not already provided in function parameters
    if None in [Ek, Emin, Emax, Ep, Epc, r, TOL]:
        print("Alternate Usage: CoreGen(Ek, Emin, Emax, Ep, Epc, r, TOL)\n\n")
        
        Ek = float(input("Edge energy (eV) = "))
        Emin = float(input("First coreloss channel (eV) = "))
        Emax = float(input("Last coreloss channel (eV) = "))
        Ep = float(input("Plasmon energy (eV) = "))
        Epc = float(input("eV/channel = "))
        r = float(input("Parameter r (e.g. 4) = "))
        TOL = float(input("thickness/inelastic-MFP = "))
    else:
        print(f"Edge energy (eV) = {Ek}")
        print(f"First coreloss channel (eV) = {Emin}")
        print(f"Last coreloss channel (eV) = {Emax}")
        print(f"Plasmon energy (eV) = {Ep}")
        print(f"eV/channel = {Epc}")
        print(f"Parameter r (e.g. 4) = {r}")
        print(f"thickness/inelastic-MFP = {TOL}")

    A = Ek ** r * np.exp(TOL)
    n = 10
    Ecore = np.arange(Emin, Emax + Epc, Epc)
    Elow = Ecore - Ek

    core = np.zeros_like(Ecore)
    low = np.zeros_like(Elow)
    
    for i in range(n):
        J = A * ((Ecore - i * Ep) ** -r) * (TOL ** i) * np.exp(-TOL) / factorial(i)
        maskcore = (Ecore - i * Ep) >= Ek
        J *= maskcore
        core += J
        
        masklow = Elow >= Ep * i
        if np.any(masklow):
            low[np.where(masklow)[0][0]] = TOL ** i / factorial(i)
    
    plt.figure()
    plt.plot(Elow, low)
    plt.title('Low-loss spectrum', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('probability')
    
    plt.figure()
    plt.plot(Ecore, core)
    plt.title('Core-loss intensity', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('core-loss intensity')
    plt.show()
    
    np.savetxt('CoreGen.cor', np.column_stack((Ecore, core)), fmt='%g %g')
    np.savetxt('CoreGen.low', np.column_stack((Elow, low)), fmt='%g %g')


if __name__ == "__main__":
    CoreGen()
