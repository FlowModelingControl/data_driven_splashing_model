# Uncertainty magnification factors for RG splashing model.
This repository contains a Python implementation of the uncertainty magnification factors (UMFs) presented in ??? for the drop impact splashing model proposed by Riboux & Gordillo (2014; 2015). Please refer to the main text for more details.

## Requirements / Dependencies
* Python 3
* numpy
* scipy
* sympy

The dependencies can be installed quickly with pip:
```
pip install -r requirements.txt
```

## Usage
For an executable version of this example see `example.py`.

Import the UMF calculator in the script where you would like to use it and create an instance of it:
```
import numpy as np
from RG2015_UMFs import RG2015_UMFs

umfs = RG2015_UMFs()
```

Create a numpy array (vector) which contains the eight impact variables for a water drop impacting a solid smooth surface in air, which are
* impact velocity in m/s,
* drop radius in m,
* liquid density in kg/m^3,
* liquid viscosity in Pa s,
* liquid surface tension in N/m,
* gas density in kg/m^3,
* gas viscosity in Pa s,
* and gas mean free path in m.

```
x1 = np.array([
    10,
    1.5e-3,
    1000,
    1e-3,
    72e-3,
    1.225,
    1.82e-5,
    68e-9
])
```

Calculate the UMFs with
```
umfs.calc_umfs(x1)
```
which yields the following output:
```
{
    'U0': 0.608923031058429, 
    'R0': 0.3001688763896971, 
    'rho_l': -0.14579179905533132, 
    'mu_l': 0.080893353004417, 
    'sigma_l': -0.43510155394908584, 
    'rho_g': 0.31961349846127896, 
    'mu_g': 0.18038650153872104, 
    'lambda_g': -0.12634739928035382
}
```

## References
* Riboux, Guillaume, and José Manuel Gordillo. "Experiments of drops impacting a smooth solid surface: a model of the critical impact speed for drop splashing." Physical review letters 113.2 (2014): 024507.
* Riboux, Guillaume, and José Manuel Gordillo. "The diameters and velocities of the droplets ejected after splashing." Journal of Fluid Mechanics 772 (2015): 630-648.
* ???
