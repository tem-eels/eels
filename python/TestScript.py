from Prism import Prism
from Sigdis import Sigdis
from SigADF import SigADF
from lenzplus import lenzplus
from SpecGen import SpecGen
from Flog import Flog
from FlogS import FlogS
from RLucy import RLucy
from Drude import Drude
from KraKro import KraKro
from Kroeger import Kroeger
from KroegerEBplots import KroegerEBplots
from CoreGen import CoreGen
from EdgeGen import EdgeGen
from Frat import Frat
from Concor2 import Concor2
from Sigmak3 import Sigmak3
from Sigmal3 import Sigmal3
from Sigpar import Sigpar
from tKKs import tKKs
from IMFP import IMFP
from PMFP import PMFP
from Afit import Afit
from Bfit import Bfit

# B.1. First-Order Spectrometer Focusing
Prism(100, 0, 45, 0.4, 10, 3, 30, 100)

# B.2. Cross Sections for Atomic Displacement and High-Angle Elastic scattering
Sigdis(6, 12, 10, 200)
SigADF(6, 12, 20, 100, 100)

# B.3. Lenz-Model Elastic and Inelastic Cross Sections
lenzplus(100, 40, 6, 10, 1.5)

# B.4. Generation of a Plural-Scattering Distribution
SpecGen(16.7, 3.2, 1, 5, 0.1, 10000, 1.5, 1000, 2, 0.5, 10)

# B.5. Fourier-Log Deconvolution
SpecGen(20, 2, 3, 6, 0.1, 10000, 0.5, 1000, 20, 0.1, 0.1)
Flog('SpecGen.psd', 0)

# B.6. Maximum-Likelihood Deconvolution
SpecGen(20, 2, 3, 6, 0.1, 10000, 0.5, 1000, 20, 0.1, 0.1)
RLucy('SpecGen.psd', '', 10)

# B.7. Drude-Model Spectrum Simulation
Drude(15, 3, 0, 0.1, 200, 5, 500, 50)

# B.8. Kramers-Kronig Analysis
Drude(15, 3, 0, 0.1, 100, 10, 1000, 50)
KraKro('Drude.ssd', 1, 100, 10, 1000, 2, 0.2)

# B.9. Krüger Simulation of Low-Loss Spectrum
Drude(15, 2, 0, 0.2, 200, 5, 600, 150)
Kroeger('Drude.eps', 200, 150, 5)
KroegerEBplots('KroegerEBplots_Si.dat',300,3,2.1)

# B.10. Core-Loss Simulation
CoreGen(232, 200, 400, 20, 0.2, 4, 0.5)
EdgeGen(232, 200, 400, 20, 0.2, 4, 0.5, 8)

# B.11. Fourier-Ratio Deconvolution
CoreGen(232, 200, 400, 20, 0.2, 4, 0.5)
Frat('CoreGen.low', 0, 'CoreGen.cor')

# B.12. Incident-Convergence Correction
Concor2(18, 12, 500, 100)

# B.13. Hydrogenic K-Shell Cross Sections
Sigmak3(6, 284, 100, 100, 10)

# B.14. Modified-Hydrogenic L-Shell Cross Sections
Sigmal3(22, 100, 80, 10)

# B.15. Parameterized K-, L-, M-, N- and O-Shell Cross Sections
Sigpar(5, 50, 'K', 100, 5)

# B.16. Measurement of Absolute Specimen Thickness
Drude(15, 3, 0, 0.2, 200, 5, 500, 100)
tKKs('Drude.ssd', 200, 0, 5, 1)

# B.17. Total-Inelastic and Plasmon Mean Free Paths
IMFP([6, 1], [12, 1], [0.33, 0.67], 5, 10, 0.9, 200)
PMFP(200, 20, 5, 10)

# B.18. Constrained Power-Law Background Fitting
Afit('CoO.dat', 500, 30, 740, 30, 'Acore.dat', 'Aback.dat')
Bfit('CoO.dat', 500, 30, 740, 30, 550, 50, 0.1, 'Bcore.dat', 'Bback.dat')
