# Unfolding Boxes with Local Constraints: CADE 30

This repository contains the implementation of the encoding for box unfoldings associated to the CADE-30 submission *"Unfolding Boxes with Local Constraints"*, by Long Qian, Eric Wang, Bernardo Subercaseaux, and Marijn Heule.

<!-- Unfolding orthogonal boxes through SAT. The folder encoder contains the 
encoding/decoder scripts. The experiments folder contains scripts used to 
generate all sub-problems for enumerating solutions. -->

## Requirements

Our code requires the Python library [PySAT](https://pysathq.github.io/), which can be installed using
```
pip install python-sat
```
For visualizing solutions, the [maptlotlib](https://matplotlib.org/) library is required, and can be installed using
```
pip install matplotlib
```
In terms of solver, this README uses [kissat](https://github.com/arminbiere/kissat/) as an example. For the enumeration of solutions, we use [allsat](https://github.com/jreeves3/allsat-cadical).

## Basic Usage
### Encoder
To find a common unfolding between boxes of dimensions $(a, b, c), (u, v, w)$ run:
```
python3 encoder.py -d a b c u v w
```
### Decoder 
If sol.cnf is the v-lines output of the encoder file + first line saying satisfiable,
the solution can be decoded by:
```
python3 decoder.py -d a b c u v w -i sol.cnf 
```
where the option `-v 1` can be added to additionally show the edges preserved.

## Enumerating all solutions
To generate all sub-problems for enumerating unfoldings between boxes of dimensions (a, b, c), (u, v, w) do:

```
python3 experiment_set_up.py -enc ../encoder/encoder.py -pair ../encoder/find_iso_pairs.py -d a b c u v w --orient2=5 -o .
```
If (a, b, c) is of the form (1, 1, n), use `--orient2=0` instead. 