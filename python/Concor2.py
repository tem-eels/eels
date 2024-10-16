import numpy as np

def Concor2(alpha=None, beta=None, e=None, e0=None):
    """
    CONCOR2: Evaluation of convergence correction F using the
    formulae of Scheinfein and Isaacson (SEM/1984, pp.1685-6).
    For absolute quantitation, divide the areal density by F2.
    For elemental ratios, divide each concentration by F2 or F1.

    ALPHA and BETA should be in mrad, E in eV, E0 in keV.
    
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    """

    print("\n---------------Concor2--------------\n\n")
    if alpha is None or beta is None or e is None or e0 is None:
        print("Alternate Usage: Concor2(alpha, beta, E, E0)\n\n")
        
        alpha = float(input("Alpha (mrad): "))
        beta = float(input("Beta (mrad): "))
        e = float(input("E (eV): "))
        e0 = float(input("E0 (keV): "))
    else:
        print(f"Alpha (mrad): {alpha}")
        print(f"Beta (mrad): {beta}")
        print(f"E (eV): {e}")
        print(f"E0 (keV): {e0}")

    tgt = e0 * (1.0 + e0 / 1022.0) / (1.0 + e0 / 511.0)
    thetae = (e + 1e-6) / tgt  # avoid NaN for e=0
    # A2, B2, T2 are squares of angles in radians^2
    a2 = alpha ** 2 * 1e-6 + 1e-10  # avoid inf for alpha=0
    b2 = beta ** 2 * 1e-6
    t2 = thetae ** 2 * 1e-6
    eta1 = np.sqrt((a2 + b2 + t2) ** 2 - 4.0 * a2 * b2) - a2 - b2 - t2
    eta2 = 2.0 * b2 * np.log(0.5 / t2 * (np.sqrt((a2 + t2 - b2) ** 2 + 4.0 * b2 * t2) + a2 + t2 - b2))
    eta3 = 2.0 * a2 * np.log(0.5 / t2 * (np.sqrt((b2 + t2 - a2) ** 2 + 4.0 * a2 * t2) + b2 + t2 - a2))
    eta = (eta1 + eta2 + eta3) / a2 / np.log(4.0 / t2)
    f1 = (eta1 + eta2 + eta3) / 2.0 / a2 / np.log(1.0 + b2 / t2)
    f2 = f1
    if alpha / beta > 1:
        f2 = f1 * a2 / b2
    bstar = thetae * np.sqrt(np.exp(f2 * np.log(1.0 + b2 / t2)) - 1.0)
    print(f"\nf1 {f1} f2 {f2} bstar {bstar}\n")
    return f1, f2, bstar

if __name__ == "__main__":
    Concor2()
