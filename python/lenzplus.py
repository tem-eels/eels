import numpy as np

def lenzplus(e0=None, e=None, z=None, beta=None, toli=None):
    """
    Calculates Lenz cross sections for elastic and inelastic scattering,
    then fractions of scattering collected by an aperture, including plural
    scattering and broadening of the elastic and inelastic angular distributions.
    
    For details, see Egerton "EELS in the EM" 3rd edition, Appendix B.
    """
    
    print("----------------------------------\n")
    
    if e0 is None or e is None or z is None or beta is None:
        e0 = float(input('Lenzplus(8-10-2010): incident energy E0(keV)= '))
        e = float(input('Ebar(eV) [Enter 0 for 6.75*Z]: '))
        z = float(input('Atomic number Z = '))
        beta = float(input('Collection semi-angle Beta (mrad)= '))
    
    if e == 0:
        e = 6.75 * z
    
    print(f"E0 (keV), Ebar (eV), Z, Beta (mrad) are: {e0}, {e}, {z}, {beta}\n")
    
    gt = 500 * e0 * (1022 + e0) / (511 + e0)
    te = e / (2 * gt)
    a0 = 0.0529
    gm = 1 + e0 / 511
    vr = np.sqrt(1 - 1 / gm ** 2)
    k0 = 2590 * gm * vr
    coeff = 4 * gm ** 2 * z / a0 ** 2 / k0 ** 4
    r0 = 0.0529 / z ** (1/3)
    t0 = 1 / (k0 * r0)
    print(f"Theta0 = {t0:.4E} rad\t\t\t\t ThetaE = {te:.4E} rad\n")
    
    b = beta / 1000
    b2 = b ** 2
    te2 = te ** 2
    t02 = t0 ** 2
    dsidom = coeff / (b2 + te2) ** 2 * (1 - t02 ** 2 / (b2 + te2 + t02) ** 2)
    dsidb = 2 * np.pi * np.sin(b) * dsidom
    
    sigin = 8 * np.pi * gm ** 2 * z ** (1/3) / k0 ** 2 * np.log((b2 + te2) * (t02 + te2) / (te2 * (b2 + t02 + te2)))
    silim = 16 * np.pi * gm ** 2 * z ** (1/3) / k0 ** 2 * np.log(t0 / te)
    f1i = sigin / silim
    
    dsedom = z * coeff / (np.sin(b / 2) ** 2 + np.sin(t0 / 2) ** 2) ** 2
    dsedb = dsedom * 2 * np.pi * np.sin(b)
    selim = 4 * np.pi * gm ** 2 * z ** (4/3) / k0 ** 2
    f1e = 1 / (1 + t02 / b2)
    sigel = f1e * selim
    
    print(f"dSe/dOmega = {dsedom:.4E} nm^2/sr\t\t\t dSi/dOmega = {dsidom:.4E} nm^2/sr")
    print(f"dSe/dBeta = {dsedb:.4E} nm^2/rad\t\t\t dSi/dBeta = {dsidb:.4E} nm^2/rad")
    print(f"Sigma (elastic) = {sigel:.4E} nm^2\t\t\t Sigma (inelastic) = {sigin:.4E} nm^2")
    print(f"F (elastic) = {f1e:.4E}\t\t\t\t\t F (inelastic) = {f1i:.4E}\n")
    
    nu = silim / selim
    print(f"Total-inelastic/total-elastic ratio = {nu:.4E}\n")
    
    if toli is None:
        toli = float(input('Enter t/lambda (inelastic) or 0 to quit: '))
    
    if toli != 0:
        print(f"t/lambda (beta) = {toli * f1i:.4E}\n")
        
        tole = toli / nu
        xe = np.exp(-tole)
        xi = np.exp(-toli)
        fie = f1e * f1i
        pun = xe * xi
        pel = (1 - xe) * xi * f1e
        pz = pun + pel
        pin = xe * (1 - xi) * f1i
        pie = (1 - xi) * (1 - xe) * fie
        pi = pin + pie
        
        print(f"p(unscat) = {pun:.4E}\t\t P(el) = {pel:.4E} neglecting elastic broadening")
        print(f"p(inel) = {pin:.4E}\t\t\t P(in+el) = {pie:.4E} neglecting inelastic broadening")
        print(f"I0/I = {pz:.4E}\t\t\t\t Ii/I = {pi:.4E} neglecting angular broadening")
        
        pt = pz + pi
        lr = np.log(pt / pz)
        print(f"ln(It/I0) = {lr:.4E} without broadening\n")
        
        f2e = 1 / (1 + 1.7**2 * t02 / b2)
        f3e = 1 / (1 + 2.2**2 * t02 / b2)
        f4e = 1 / (1 + 2.7**2 * t02 / b2)
        pe = xe * (tole * f1e + tole**2 * f2e / 2 + tole**3 * f3e / 6 + tole**4 * f4e / 24)
        peni = pe * xi
        pu = xi * xe
        rz = pu + peni
        
        pi = xi * (np.exp(toli * f1i) - 1)
        pine = xe * pi
        pie = pi * pe
        ri = pine + pie
        
        print(f"P(unscat) = {pu:.4E}\t\t P(el only) = {peni:.4E} with elastic broadening")
        print(f"P(inel only) = {pine:.4E}\t\t P(in+el) = {pie:.4E} with inelastic broadening")
        print(f"I0/I = {rz:.4E}\t\t\t\t Ii/I = {ri:.4E} with angular broadening")
        
        rt = rz + ri
        lr = np.log(rt / rz)
        print(f"ln(It/I0) = {lr:.4E} with angular broadening\n")

# Example usage:
# lenzplus(200, 100, 13, 50, 0.5)
if __name__ == "__main__":
    lenzplus()