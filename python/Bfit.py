import numpy as np
import matplotlib.pyplot as plt
from ReadData import ReadData

def Bfit(infile=None, eprestart=None, eprewidth=None, epoststart=None, epostwidth=None, ecorestart=None, ecorewidth=None, differ=None, outcore=None, outback=None):
    print("\n------------Bfit-----------\n\n")

    # Read spectrum from input file
    if infile is None:
        print("Alternate Usage: Bfit('infile', eprestart, eprewidth, epoststart, epostwidth, ecorestart, ecorewidth, differ, outcore, outback)\n\n")
        infile = input("Input file: ")
    else:
        print(f"Input file: {infile}")
    
    spectrum = ReadData(infile,2).T

    edispersion = spectrum[0, 2] - spectrum[0, 1]
    print(f"\nEnergy dispersion {edispersion} [eV/ch]\n")

    # Acquire pre/post/core energy window settings (if not in input parameters)
    if eprestart is None:
        eprestart = float(input("Pre-Edge energy window START [eV]: "))
        eprewidth = float(input("Pre-Edge energy window WIDTH [eV]: "))
        epoststart = float(input("Post-Edge energy window START [eV]: "))
        epostwidth = float(input("Post-Edge energy window WIDTH [eV]: "))
        ecorestart = float(input("Core-loss energy window START [eV]: "))
        ecorewidth = float(input("Core-loss energy window WIDTH [eV]: "))
        differ = float(input("Convergence criteria in percent difference in R in consequentive loops: "))
    else:
        print(f"Pre-Edge energy window START [eV]: {eprestart}")
        print(f"Pre-Edge energy window WIDTH [eV]: {eprewidth}")
        print(f"Post-Edge energy window START [eV]: {epoststart}")
        print(f"Post-Edge energy window WIDTH [eV]: {epostwidth}")
        print(f"Core-loss energy window START [eV]: {ecorestart}")
        print(f"Core-loss energy window WIDTH [eV]: {ecorewidth}")
        print(f"Convergence criteria in percent difference in R in consequentive loops: {differ}")

    epreend = eprestart + eprewidth
    epremid = eprestart + eprewidth / 2
    epostend = epoststart + epostwidth
    ecoreend = ecorestart + ecorewidth

    # Define energy windows as logical array indices
    prefilter = (spectrum[0, :] >= eprestart) & (spectrum[0, :] < epreend)
    pre1filter = (spectrum[0, :] >= eprestart) & (spectrum[0, :] < epremid)
    pre2filter = (spectrum[0, :] >= epremid) & (spectrum[0, :] < epreend)
    corefilter = (spectrum[0, :] >= ecorestart) & (spectrum[0, :] < ecoreend)
    postfilter = (spectrum[0, :] >= epoststart) & (spectrum[0, :] < epostend)

    # Calculate sums of Pre, Post and Core energy windows as defined above
    ipre = np.sum(spectrum[1, prefilter]) * edispersion
    ipre1 = np.sum(spectrum[1, pre1filter]) * edispersion
    ipre2 = np.sum(spectrum[1, pre2filter]) * edispersion
    icore = np.sum(spectrum[1, corefilter]) * edispersion
    ipost = np.sum(spectrum[1, postfilter]) * edispersion

    # Calculate initial values of A and R
    print(f"\nIpre1 {ipre1}, Ipre2 {ipre2}, Ipre {ipre}, di {ipre - (ipre1 + ipre2)}")
    r = 2.0 * np.log(ipre1 / ipre2) / np.log(epreend / eprestart)
    a = ipre * (1 - r) / (epreend ** (1 - r) - eprestart ** (1 - r))
    print(f"First estimates of R = {r}, A = {a}\n")

    # Iterate until last two calculations of R are within defined convergence criteria
    for loop in range(30):
        print(f"+++++++++++++ iteration loop # {loop + 1} +++++++++++++ ")
        r_prev_loop = r
        icoreback = a / (1 - r) * (ecoreend ** (1 - r) - ecorestart ** (1 - r))
        b = (icore - icoreback) * (1 - r) / (ecoreend ** (1 - r) - ecorestart ** (1 - r))  # constant B for the core intensity
        ipostcore = b * (epostend ** (1 - r) - epoststart ** (1 - r)) / (1 - r)  # calculating the CORE contribution in the post-edge region
        ipostback = ipost - ipostcore
        r = 2.0 * np.log(ipre * epostwidth / (ipostback * eprewidth)) / np.log(epoststart * epostend / (eprestart * epreend))
        a = ipre * (1 - r) / (epreend ** (1 - r) - eprestart ** (1 - r))
        print(f"POST EDGE WINDOW IpostCore = {ipostcore}")
        print(f"CORE WINDOW Measured = {icore}")
        print(f"Exponent R = {r}, background A = {a}")
        if (r / r_prev_loop > 1 - differ / 100) and (r / r_prev_loop < 1 + differ / 100):
            print(f"END change in R less than {(1 - r / r_prev_loop) * 100} percent in last two loops")
            break

    print('Fit coefficients I(E) = A*E^r')
    print(f"r = {-r}, A = {a}")

    # Generate core and background curve data from A and R 
    back = spectrum.copy()
    back[1, :] = a * spectrum[0, :] ** -r
    core = spectrum.copy()
    core[1, :] = spectrum[1, :] - back[1, :]
    backfilt = (back[1, :] > 0) & (back[0, :] > 20)
    corefilt = (core[0, :] >= epreend) & (core[1, :] > 0)
    back = back[:, backfilt]
    core = core[:, corefilt]

    # Plot spectrum, core, and fitted curve
    plt.figure()
    plt.plot(spectrum[0, :], spectrum[1, :], 'k', linewidth=2, label='Spectrum')
    plt.plot(core[0, :], core[1, :], 'g', linewidth=2, label='Core')
    plt.plot(back[0, :], back[1, :], 'r', linewidth=2, label='BFit')
    plt.title('BFit', fontsize=12)
    plt.xlabel('Energy Loss [eV]')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    # Ask for output file names if none provided with input parameters
    if outcore is None:
        outcore = input("Filename for core: ")
    else:
        print(f"Filename for core: {outcore}")

    if outback is None:
        outback = input("Filename for background: ")
    else:
        print(f"Filename for background: {outback}")

    # Save output to files
    np.savetxt(outcore, core.T, fmt='%f')
    np.savetxt(outback, back.T, fmt='%f')

if __name__ == '__main__':
    Bfit()