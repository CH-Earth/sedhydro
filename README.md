# SedHydro
SedHydro is a hydrological-model-coupled, multi-fraction erosion and sediment transport framework designed for simulation of erosion, transport, deposition, and re-entrainment of sediment across river basins of any scale. The framework is modular and hydrology-model agnostic, allowing coupling with any hydrological model capable of providing the required hydrological forcing and river hydraulic variables. The framework has been developed to provide a flexible and computationally scalable environment to coupling externally simulated hydrology with spatially distributed erosion processes and physically based river sediment routing. 
<p align="center">
  <img src="docs/figures/erosion_scheme.jpg" width="500">
</p>


The framework was developed to address several limitations commonly encountered in hydro-
sediment modelling systems, including:

- poor representation of geomorphic processes within the river network, which strongly affects sediment transport,
- limited representation of hillslope sediment connectivity,
- inability to separately route sediment fractions,
- insufficient representation of cold-region processes,
- strong dependency on a specific hydrological model structure,
- and computational limitations for large river basins and long simulation periods.

SedHydro integrates multiple sediment-process components into a unified modular framework in-
cluding:

- spatially distributed erosion generation,
- hillslope sediment delivery routing,
- physical-based river sediment transport,
- multi-fraction sediment dynamics,
- and depressional storage attenuation.

<p align="center">
  <img src="docs/figures/framework_coupling.jpeg" width="800">
</p>

## Repository structure
```
SedHydro/
│
├── SedHydro.py
├── SedHydro_mp.py
├── optimisation_updated.py
├── utils.py
├── mapMaker.py
├── directory_settings_*.toml
├── environment.yml
│
├── TempSedRout/
│   ├── TempSedRout_function.py
│   ├── TempSedRout_storage_function.py
│   └── constants.toml
│
├── outputs/
├── settings/
├── shapefiles/
├── attributes/
└── data/
```

## Installation

See: 
**[SedHydro_Installation_Guide.md](SedHydro_Installation_Guide.md)**

## Citation
Citation information will be added following publication of the SedHydro framework.
