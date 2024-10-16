import numpy as np
import matplotlib.pyplot as plt

def SpecGen(ep=None, wp=None, wz=None, ez=None, epc=None, a0=None, tol=None, nd=None, back=None, fback=None, cpe=None):
    """
    SpecGen: Generates a plural-scattering distribution from a Gaussian-shaped
    single scattering distribution (SSD) of width wp, peaked at ep,
    with background and Poisson shot noise.
    
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011.
    """

    print("   SpecGen(ep, wp, wz, ez, epc, a0, tol, nd, back, fback, cpe)\n")

    if ep is None or wp is None or wz is None or ez is None or epc is None or a0 is None or tol is None or nd is None or back is None or fback is None or cpe is None:
        ep = float(input('SpecGen: plasmon energy (eV) = '))
        wp = float(input('plasmon FWHM (eV) : '))
        wz = float(input('zero-loss FWHM (eV): '))
        ez = float(input('zero-loss offset from first channel (eV): '))
        epc = float(input('eV/channel: '))
        a0 = float(input('zero-loss counts: '))
        tol = float(input('t/lambda: '))
        nd = int(input('number of channels: '))
        back = float(input('instrumental background level (counts/channel): '))
        fback = float(input('instrumental noise/background (e.g. 0.1): '))
        cpe = float(input('spectral counts per beam electron (e.g. 0.1): '))
    else:
        print(f'SpecGen: plasmon energy (eV) = {ep}')
        print(f'plasmon FWHM (eV) : {wp}')
        print(f'zero-loss FWHM (eV): {wz}')
        print(f'zero-loss offset from first channel (eV): {ez}')
        print(f'eV/channel: {epc}')
        print(f'zero-loss counts: {a0}')
        print(f't/lambda: {tol}')
        print(f'number of channels: {nd}')
        print(f'instrumental background level (counts/channel): {back}')
        print(f'instrumental noise/background (e.g. 0.1): {fback}')
        print(f'spectral counts per beam electron (e.g. 0.1): {cpe}')

    print('-------------------------------')
    fpoiss = np.sqrt(cpe)
    sz = wz / 1.665  # convert from FWHM to standard deviation
    sp = wp / 1.665
    hz = a0 / sz / 1.772  # height of ZLP

    psd = np.zeros(nd)
    ssd = np.zeros(nd)
    outssd = np.zeros(nd)
    outpsd = np.zeros(nd)
    eout = np.zeros(nd)
    
    rlnum = 1.23456
    
    with open('SpecGen.ssd', 'w+') as fid_1, open('SpecGen.psd', 'w+') as fid_2:
        for i in range(nd):
            e = (i + 1) * epc - ez
            fac = 1
            psd[i] = 0

            for order in range(15):
                sn = np.sqrt(sz**2 + order * sp**2)
                xpnt = (e - order * ep)**2 / sn**2
                if xpnt > 20.0:
                    expo = 0.0
                else:
                    expo = np.exp(-xpnt)

                dne = hz * sz / sn * expo / fac * tol**order
                rndnum = 2 * (int(rlnum) - rlnum)
                snoise = fpoiss * (np.sqrt(dne) * rndnum)
                rlnum = 9.8765 * rndnum

                if order == 1:
                    bnoise = fback * back * rndnum
                    ssd[i] = dne + np.sqrt(snoise**2 + bnoise**2)
                    outssd[i] = ssd[i] + back
                    fid_1.write(f'{e:.15g} {ssd[i] + back:.15g}\n')
                
                psd[i] += dne
                fac *= (order + 1)

            snoise = fpoiss * (np.sqrt(psd[i]) * rndnum)
            outpsd[i] = psd[i] + np.sqrt(snoise**2 + bnoise**2) + back
            fid_2.write(f'{e:.15g} {outpsd[i]:.15g}\n')
            eout[i] = e
    
    plt.figure()
    plt.plot(eout, outssd, 'b', label='SSD')
    plt.plot(eout, outpsd, ':r', label='PSD')
    plt.legend()
    plt.title('SpecGen', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('Counts/channel')
    plt.show()

if __name__ == "__main__":
    SpecGen()