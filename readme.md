# PyTEDI

**PyTEDI is a hydrological tool for estimating the impacts of small dams on streamflow.**

**PyTEDI** stands for the **Py**thon **T**ool for **E**stimating **D**am **I**mpacts.

Small dams are sometimes known as 'farm dams', 'farm ponds', 'small artificial impoundments', 'farm storages', 'small storages', 'tanks', 'stock ponds', or 'mill ponds'. They are known to have a significant impact on streamflow in downstream waterways.

PyTEDI was developed in 2026 by [Alluvium](https://alluvium.com/) and [FlowMatters](https://www.flowmatters.com.au/), and was commissioned by the Victorian [Department of Energy Environment and Climate Action](https://www.deeca.vic.gov.au/).

## Description

PyTEDI estimates the impact of small dams on streamflow in a catchment. A water balance is calculated for every timestep (monthly or daily) and every dam in the model based on:

- flow recorded downstream (or upstream) of the dams
- rainfall and evaporation data
- on-site water extraction patterns
- peak flow rate and seasonal operation dates of low flow bypasses for each dam
- peak flow rate and seasonal operation dates of winterfill pumping for each dam
- details of the size, volume, and spatial arrangement of the dams

The water balance calculation produces the following outputs for each individual dam and aggregated across the catchment:

- flow upstream of the dam(s)
- flow downstream of the dam(s)
- net evaporation (evaporation less rainfall) from the surface of the dam(s) *(both potential and actual)*
- on-site extractions from the dam(s) *(both potential and actual)*
- low flow bypass volumes *(both potential and actual)*
- winterfill extraction volumes *(both potential and actual)*
- volume of water stored in each dam at the start and end of each timestep
- dam spills

## Capabilities
PyTEDI is a hydrological modelling tool which is intended to work within Python workflows. The key features of pyTEDI include:

- **Backward compatibility** - existing model files from the older Windows-based STEDI farm dam tool can be read and interpreted
- **Computational speed** - use of the Python Numba library means that calculations are lightning fast
- **Seamless integration with Python** - all model inputs and outputs can be passed directly to and from Python via pandas or other standard Python data types. Writing input and output files is entirely optional.
- **Two historical scaling options** - models can be scaled by individual dam construction date, or by global percentages applied to historical dates.

## History

TEDI (later renamed STEDI) was originally developed in the late 1990s by the Victorian government, the Murray Darling Basin Commission, and Sinclair Knight Merz (now Jacobs). It was published as a Windows executable with a full graphical user interface. It has been through many incarnations and used for many purposes. The latest Windows version is be available from Jacobs' website (although the download button was not working as of November 2026): https://www.jacobs.com/natural-resource-management

PyTEDI is intended as a full Python-based replacement for STEDI to improve workflow efficiency and reduce scope for manual errors.

## Requirements

PyTEDI has very mild operating requirements:

- Python 3.11 or later
- Pandas
- NumPy
- Numba (needed to make the code go really really fast)
- Tomllib (needed to read and write model parameter files)

## Installation

Install pyTEDI from the Python Package Index (PyPI) by opening a command prompt or terminal and navigating to your selected Python environment, then running:

```bash
pip install pytedi
```

If that doesn't work, maybe pip is not recognised. You could also try:

```bash
python -m pip install pytedi
```

If you really know what you are doing, you can download the wheel file directly from this repository and install it. But only if you are sure you are a Python wizard. Pip is much easier.

## Example usage

For more detailed information about how to use pyTEDI, please read the [user manual](UserManual.md).

To open and run an existing model file (.scn) created using the old Windows-based STEDI tool, you would use the following code:

```python
import pytedi

scn_file = r'data\Example_STEDI_scenario.scn'                # where is the model file?
mymodel = pytedi.open_windows_STEDI_scenario(scn_file)       # read the model file
mymodel.run_model()                                          # run the model
mymodel.catch_results.head()                                 # show the first few lines of the results
```
In each Python session, the first time the `run_model` command is called there will be a short delay of 1 to 5 seconds while the tool self-compiles a few short sections of code. This is normal. Every subsequent run in that Python session will be much, much faster.

Please refer to the [user manual](UserManual.md) for examples of how to:

- save results into a csv file for the overall catchment
- save results into a csv file for an individual dam
- save a model into native pyTEDI format (a combination of csv and toml formats)
- open a model in native pyTEDI format
- integrate results into eWater Source
- run a model with historical scaling

## US data (optional)

The optional `pytedi.us` subpackage builds a dam list for any area in the United States from public data: USGS 3DEP terrain, USGS National Hydrography Dataset (NHD) ponds and streams, the USACE National Inventory of Dams, and farm ponds detected in Sentinel-2 imagery. It is not needed for the core model and is installed separately:

```
pip install "pytedi[us]"
```

```python
from pytedi.us import build_damlist, huc12_at

result = build_damlist(huc12_at(37.09, -79.30))   # the HUC12 watershed around Hurt, Virginia
model = result.to_model()                          # a tedimodel with the dam list and catchment area
result.damlist.head()
```

See [`examples/us_hurt_va`](examples/us_hurt_va/README.md) for a full example, the method, and its limits.

## Licence

This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

In short:
- Feel free to use, modify and share the software.
- Keep the original copyright and licence notice with copies or substantial portions of the software.
- The software is provided "as is", without warranty.
- Please give credit where credit is due.

[![Licence: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)

## Credits

PyTEDI was developed in 2026 by [Alluvium](https://alluvium.com/) and [FlowMatters](https://www.flowmatters.com.au/), and was commissioned by the Victorian [Department of Energy Environment and Climate Action](https://www.deeca.vic.gov.au/).

The pyTEDI team includes:

- Robert Morden (Alluvium)
- Ella Guthrie (Alluvium)
- Joel Rahman (FlowMatters)
- Diana Dai (Alluvium)

We acknowledge the financial and collaborative support provided by the Surface Water Assessment and Modelling team at DEECA.


## References

The core pyTEDI calculation algorithm is based on the original paper by Nathan et al. (2005), with several subsequent papers extending the data input methodology and calculation methods.

- Nathan, R., Jordan, P., Morden, R., 2005. Assessing the impact of farm dams on streamflows, Part I: Development of Simulation Tools. *Australian Journal of Water Resources* 9, 1–12. https://doi.org/10.1080/13241583.2005.11465259
- Nathan, R., Lowe, L., 2012. The hydrologic impacts of farm dams. *Australasian Journal of Water Resources* 16, 75–83. https://doi.org/10.7158/13241583.2012.11465405
- Fowler, K., Morden, R., Lowe, L., Nathan, R., 2015. Advances in assessing the impact of hillside farm dams on streamflow. *Australian Journal of Water Resources* 19. https://doi.org/10.1080/13241583.2015.1116182
- Morden, R., 2017. A new method of accounting for runoff dams, in: Syme, G., Hatton MacDonald, D., Fulton, B., Piantadosi, J. (Eds.), *MODSIM2017, 22nd International Congress on Modelling and Simulation*. Hobart, Australia, pp. 1676–1682. https://doi.org/10.36334/MODSIM.2017.L8.morden
