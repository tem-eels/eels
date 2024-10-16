import numpy as np
from ReadData import ReadData

def Sigpar(z=None, dl=None, shell=None, e0=None, beta=None):
    """
    SIGPAR2.FOR calculates sigma(beta,delta) from f-values stored
    in files FK.DAT, FL.DAT, FM45.DAT, FM23.DAT, and FNO45.DAT
    based on values given in Ultramicroscopy 50 (1993) 13-28.
    """
    
    print('\n---------------Sigpar----------------\n\n')

    if z is None or dl is None or shell is None:
        print('Alternate Usage: Sigpar(Z, Delta, Shell, E0, Beta)\n\n')
        z = int(input('Enter Z: '))
        dl = float(input('Enter Delta (eV): '))
        shell = input('Enter edge type (K,L,M23,M45,N or O): ').upper()
    else:
        print(f'Z: {z}')
        print(f'Delta (eV): {dl}')
        print(f'Edge type (K,L,M23,M45,N or O): {shell.upper()}')

    # Select f-values table based on edge type
    if shell == 'K':
        infile = 'Sigpar_fk.dat'
        num_col = 6
    elif shell == 'L':
        infile = 'Sigpar_fl.dat'
        num_col = 6
    elif shell == 'M23':
        infile = 'Sigpar_fm23.dat'
        num_col = 3
    elif shell == 'M45':
        infile = 'Sigpar_fm45.dat'
        num_col = 6
    elif shell in ['N', 'O']:
        infile = 'Sigpar_fno45.dat'
        num_col = 5
    else:
        raise ValueError(f"Invalid Edge Type '{shell}'")

    # Read edge type data
    in_data = ReadData(infile, num_col)

    # Lookup Z value in edge type table
    fdata = in_data[in_data[:, 0] == z]
    if fdata.size == 0:
        raise ValueError(f'Z value {z} not found for edge type {shell} ({infile})')

    # Get f-values from table
    if shell in ['K', 'L', 'M45']:
        ec = fdata[0, 1]
        f50 = fdata[0, 2]
        f100 = fdata[0, 3]
        f200 = fdata[0, 4]
        erp = fdata[0, 5]
        fd = fdcalc(dl, f50, f100, f200)
    elif shell == 'M23':
        ec = fdata[0, 1]
        f30 = fdata[0, 2]
        dl = 30
        fd = f30
        erp = 10
        print('For delta = 30eV\n')
    elif shell in ['N', 'O']:
        ec = fdata[0, 1]
        f50 = fdata[0, 2]
        f100 = fdata[0, 3]
        erp = fdata[0, 4]
        fd = fdcalc(dl, f50, f100, f100)

    # Get e0 and beta
    print(f'Ec = {ec:.15g} eV,  f(delta) =  {fd:.15g} \n')
    if e0 is None or beta is None:
        e0 = float(input('Enter E0(keV): '))
        beta = float(input('Enter beta(mrad): '))
    else:
        print(f'E0(keV): {e0}')
        print(f'beta(mrad): {beta}')

    if (beta**2) > (50 * ec / e0):
        print('Dipole Approximation NOT VALID, sigma will be too high!\n')

    # Calculate Sigma
    ebar = np.sqrt(ec * (ec + dl))
    gamma = 1 + e0 / 511
    g2 = gamma**2
    v2 = 1 - 1 / g2
    b2 = beta**2
    thebar = ebar / e0 / (1 + 1 / gamma)
    t2 = thebar**2
    gfunc = np.log(g2) - np.log((b2 + t2) / (b2 + t2 / g2)) - v2 * b2 / (b2 + t2 / g2)
    squab = np.log(1 + b2 / t2) + gfunc
    sigma = 1.3e-16 * g2 / (1 + gamma) / ebar / e0 * fd * squab
    print(f'sigma = {sigma:.3g} cm^2; \n')
    if not (beta**2 > 50 * ec / e0):
        print(f'estimated accuracy = {erp:.4g} % \n')


def fdcalc(dl, f50, f100, f200):
    if dl <= 50:
        fd = f50 * dl / 50
    elif 50 < dl < 100:
        fd = f50 + (dl - 50) / 50 * (f100 - f50)
    elif 100 <= dl < 250:
        fd = f100 + (dl - 100) / 100 * (f200 - f100)
    return fd

if __name__ == "__main__":
    Sigpar()