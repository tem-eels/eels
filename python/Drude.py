import numpy as np
import matplotlib.pyplot as plt

def Drude(ep=None, ew=None, eb=None, epc=None, e0=None, beta=None, nn=None, tnm=None):
    """
    Given the plasmon energy (ep), plasmon FWHM (ew), and binding energy (eb), 
    this program generates:
    EPS1, EPS2 from modified Eq. (3.40), ELF=Im(-1/EPS) from Eq. (3.42),
    single scattering from Eq. (4.26) and SRFINT from Eq. (4.31)
    The output is e, ssd into the file Drude.ssd (for use in Flog etc.) 
    and e, eps1, eps2 into Drude.eps (for use in Kroeger etc.)
    Gives probabilities relative to zero-loss integral (I0 = 1) per eV
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    Version 10.11.26
    """
    
    print("   Drude(ep, ew, eb, epc, e0, beta, nn, tnm)")
    if ep is None or ew is None or eb is None or epc is None or e0 is None or beta is None or nn is None or tnm is None:
        ep = float(input("plasmon energy (eV): "))
        ew = float(input("plasmon width (eV): "))
        eb = float(input("binding energy (eV), 0 for a metal: "))
        epc = float(input("eV per channel: "))
        e0 = float(input("incident energy E0 (keV): "))
        beta = float(input("collection semiangle beta (mrad): "))
        nn = int(input("number of data points: "))
        tnm = float(input("thickness (nm): "))
    else:
        print(f"plasmon energy (eV): {ep}")
        print(f"plasmon width (eV): {ew}")
        print(f"binding energy (eV): {eb}")
        print(f"eV per channel: {epc}")
        print(f"incident energy E0 (keV): {e0}")
        print(f"collection semiangle beta (mrad): {beta}")
        print(f"number of data points: {nn}")
        print(f"thickness (nm): {tnm}")
    
    b = beta / 1000.0  # rad
    T = 1000.0 * e0 * (1.0 + e0 / 1022.12) / (1.0 + e0 / 511.06) ** 2  # eV
    tgt = 1000.0 * e0 * (1022.12 + e0) / (511.06 + e0)  # eV
    rk0 = 2590.0 * (1.0 + e0 / 511.06) * np.sqrt(2.0 * T / 511060.0)
    
    e = epc * np.arange(1, nn)
    eps = 1 - ep ** 2 / (e ** 2 - eb ** 2 + e * ew * 1j)
    eps1 = np.real(eps)
    eps2 = np.imag(eps)
    elf = ep ** 2 * e * ew / ((e ** 2 - ep ** 2) ** 2 + (e * ew) ** 2)
    rereps = eps1 / (eps1 ** 2 + eps2 ** 2)
    the = e / tgt  # varies with energy loss!
    srfelf = np.imag(-4 / (1 + eps)) - elf  # for 2 surfaces
    angdep = np.arctan(b / the) / the - b / (b ** 2 + the ** 2)
    srfint = angdep * srfelf / (3.1416 * 0.0529 * rk0 * T)  # probability per eV
    anglog = np.log(1.0 + b ** 2 / the ** 2)
    volint = tnm / (3.1416 * 0.0529 * T * 2.0) * elf * anglog  # probability per eV
    ssd = volint + srfint
    
    np.savetxt('Drude.ssd', np.column_stack((e, ssd)), fmt='%0.15g %0.15g')
    np.savetxt('Drude.eps', np.column_stack((e, eps1, eps2)), fmt='%0.15g %0.15g %0.15g')
    
    Ps = np.trapz(srfint, e)  # 2 surfaces but includes negative begrenzungs contribution.
    Pv = np.trapz(volint, e)  # integrated volume probability
    lam = tnm / Pv  # does NOT depend on free-electron approximation (no damping). 
    lamfe = 4.0 * 0.05292 * T / ep / np.log(1 + (b * tgt / ep) ** 2)  # Eq.(3.44) approximation
    print(f'Ps (2 surfaces + begrenzung terms) = {Ps}, Pv = t/lambda(beta) = {Pv}')
    print(f'Volume-plasmon MFP (nm) = {lam}, Free-electron MFP (nm) = {lamfe}')
    print('--------------------------------')
    
    # Plot E, EPS1, EPS2, bulk and surface energy-loss functions
    plt.figure()
    plt.plot(e, eps1, 'r', label='eps1')
    plt.plot(e, eps2, 'g', label='eps2')
    plt.plot(e, elf, 'k', label='Im[-1/eps]')
    plt.plot(e, srfelf, 'b', label='Im[(-4/(1+eps)]')
    plt.plot(e, rereps, 'm', label='Re[1/eps]')
    plt.legend()
    plt.title('Drude dielectric data', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    yScaleMax = elf[np.where(e >= ep)[0][0]] * 2
    plt.ylim([-yScaleMax, yScaleMax])
    plt.show(block=False)
    
    # Plot volume, surface, and total intensities
    plt.figure()
    plt.plot(e, volint, 'r', label='volume')
    plt.plot(e, srfint, 'g', label='surface')
    plt.plot(e, ssd, 'b', label='total')
    plt.legend()
    plt.title('Drude probabilities', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('dP/dE [/eV]')
    plt.show()

if __name__ == "__main__":
    Drude()