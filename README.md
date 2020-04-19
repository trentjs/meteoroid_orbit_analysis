# Meteoroid Orbit Analysis

Various meteoroid orbit determination algorithms developed for the DFN as part of my PhD.


## Meteoroid Entry Orbit

The entryOrbit module is used to accurately determine the orbit of a meteoroid before entering Earth's graviational well.

For general intruction / help, run:
```
python3 entry_orbit.py -h
```

For a single orbit determination, run:
```
python3 entry_orbit.py -i data/DN161031_01_triangulation_all_timesteps.ecsv -O integrate_EOE -p
```

For a monte-carlo orbit analysis, run:
```
python3 entry_orbit.py -i data/DN161031_01_triangulation_all_timesteps.ecsv -O integrate_posvel -p -n 100
```


## Meteoroid Historical Orbit

The OrbitRegressionAnalysis module is used to probabilistically determine the likely orbital history of the observed meteoroid.

For general intruction / help, run:
```
python3 OrbitalRegressionAnalysis.py -h
```

For a monte-carlo orbit analysis, run:
```
mpirun -n 4 python3 OrbitalRegressionAnalysis.py -e data/DN161031_01_triangulation_all_timesteps.ecsv -n 100 -yr 10000 -o 1000
```
