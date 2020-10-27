# Data-driven splashing threshold model for drop impact on dry smooth surfaces
This repository provides the Python implementation for the uncertainty quantification method and the data-driven splashing threshold (DST) model presented by Pierzyna et al. (2020). 

* The uncertainty quantifier (see `example_uq.py`) propagates user-definable measurement uncertainties through the equations of the Riboux & Gordillo (2014; 2015) splashing model. It yields the combined uncertainty of the RG splashing parameter `beta` (relative or absolute) for a given set of measured drop impacts.
* The DST model (see `example_dst.py`) can be used to calculate the splashing threshold for drop impacts on a dry smooth surface and to predict the respective splashing outcome (deposition or splashing). The threshold was derived using sophisticated machine learning techniques for a wide range of impact conditions as detailed in Pierzyna et al. (2020) and is based on the RG splashing model (Riboux & Gordillo 2014; 2015).

Please refer to our article for more details (Pierzyna et al. 2020).

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
Please refer to `example_dst.py` and `example_uq.py` for detailed instructions on how to use the DST model and the uncertainty quantification method.

In general, all functions expect a vector (`numpy.ndarray` of shape `(8, )`) or a list of vectors (`numpy.ndarray` of shape `(n, 8`)) which describe the full state of the drop impact on a dry smooth surface. The expected variables are 
* impact velocity `V0` in m/s,
* drop **radius** `R0` in m,
* liquid density `rho_l` in kg/m^3,
* liquid viscosity `mu_l` in Pa s,
* liquid surface tension `sigma_l` in N/m,
* gas density `rho_g` in kg/m^3,
* gas viscosity `mu_g` in Pa s,
* and gas mean free path `lambda_g` in m.

## Testing
Tests are provided in the `tests` folder to ensure that all mathematical models are implemented correctly. Reference values were caluculated with great care in Mathematica based on equations provided by Pierzyna et al. (2020) and Riboux & Gordillo (2014; 2015).

Following command runs the tests in your terminal and should exit without erros:
```
python -m unittests
```

## References
* Pierzyna, Maximilian, David A. Burzynski, Stephan E. Bansmer, and Richard Semaan. "Data-driven splashing threshold model for drop impact on dry smooth surfaces." Journal of Fluid Mechanics (2020, submitted)
* Riboux, Guillaume, and José Manuel Gordillo. "Experiments of drops impacting a smooth solid surface: a model of the critical impact speed for drop splashing." Physical review letters 113.2 (2014): 024507.
* Riboux, Guillaume, and José Manuel Gordillo. "The diameters and velocities of the droplets ejected after splashing." Journal of Fluid Mechanics 772 (2015): 630-648.
