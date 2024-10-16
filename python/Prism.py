import numpy as np
def eq8(xy, u):
    m8 = np.eye(4)
    m8[0, 1] = u
    m8[2, 3] = u
    xy = np.dot(m8, xy)
    # print('m8 =', m8)
    # print('xy8 =', xy)
    return xy
def eq9(xy, eps, K1, g, R):
    psi = (g / R) * K1 * (1 + np.sin(np.deg2rad(eps)) ** 2) / np.cos(np.deg2rad(eps))
    psid = psi * 180 / np.pi # convert to degrees
    m9 = np.eye(4)
    m9[1, 0] = np.tan(np.deg2rad(eps)) / R
    m9[3, 2] = -np.tan(np.deg2rad(eps - psid)) / R
    xy = np.dot(m9, xy)
    return xy
def eq11(xy, phi, R):
    m11 = np.eye(4)
    m11[0, 0] = np.cos(np.deg2rad(phi))
    m11[1, 0] = -np.sin(np.deg2rad(phi)) / R
    m11[0, 1] = R * np.sin(np.deg2rad(phi))
    m11[1, 1] = np.cos(np.deg2rad(phi))
    m11[2, 3] = phi * R * np.pi / 180
    xy = np.dot(m11, xy)
    return xy
def Prism(u=None, eps1=None, eps2=None, K1=None, g=None, R=None,
phi=None, v=None):
    print('\n-----------------Prism---------------\n\n')
    if any(arg is None for arg in (u, eps1, eps2, K1, g, R, phi, v)):
        print('Alternate Usage: Prism( u, eps1, eps2, K1, g, R, phi, v )\n\n')
        u = float(input('Object distance u : '))
        eps1 = float(input('Entrance tilt epsilon1 (deg): '))
        eps2 = float(input('Exit tilt epsilon2 (deg): '))
        K1 = float(input('Fringing-field parameter K1 (e.g. 0 or 0.4): '))
        g = float(input('Polepiece gap g : '))
        R = float(input('Bend radius R : '))
        phi = float(input('Bend angle phi (deg) : '))
        v = float(input('Image distance v : '))
    else:
        print('Object distance u :', u)
        print('Entrance tilt epsilon1 (deg):', eps1)
        print('Exit tilt epsilon2 (deg):', eps2)
        print('Fringing-field parameter K1 (e.g. 0 or 0.4):', K1)
        print('Polepiece gap g :', g)
        print('Bend radius R :', R)
        print('Bend angle phi (deg) :', phi)
        print('Image distance v :', v)
    x0 = 0
    y0 = 0
    dx0 = 0.001 # 1 mrad entrance
    dy0 = 0.001 # 1 mrad entrance
    xy0 = np.array([x0, dx0, y0, dy0])
    xy = eq8(xy0, u)
    xy = eq9(xy, eps1, K1, g, R) # calculate for eps1
    xy = eq11(xy, phi, R) # magnetic field
    xy = eq9(xy, eps2, K1, g, R) # calculate for eps2
    fl = -xy[0] / xy[1]
    print('Entrance-cone semi-angle = 1 mrad.\n\n')
    print('For v =', v)
    ans1 = eq8(xy, v)
    print('x =', ans1[0])
    print("x' =", ans1[1])
    print('y =', ans1[2])

    print("y' =", ans1[3], "\n")
    print('For v =', fl)
    ans2 = eq8(xy, fl)
    print('x =', ans2[0])
    print("x' =", ans2[1])
    print('y =', ans2[2])
    print("y' =", ans2[3])

if __name__ == "__main__":
    Prism()