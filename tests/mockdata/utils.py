
import numpy as np

def mock_etof_calibration_constants():
    return 623419.734, 946.026, 11.527

def mock_etof_mono_energies():
    # 200 trains with 10 energies and 20 trains per energy
    energy = list()
    for e in np.linspace(970.0, 1060.0, 10):
        energy += [e]*20
    energy = np.array(energy)
    return energy
