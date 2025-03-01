# Unfolding Boxes with Local Constraints: CADE-30

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
To find a common unfolding between boxes of dimensions $(a, b, c), (u, v, w)$ run:
```
python3 encoding/encoder.py -d <a> <b> <c> <u> <v> <w>
```
As a result, the file `common_unfolding.cnf` will be generated.

For example, if we run
```
python3 encoding/encoder.py -d 1 1 5 1 2 3
kissat common_unfolding.cnf > solution.txt
```
A solution is obtained in about 1 second. To decode the solution, run
```
python3 encoding/decoder.py -d 1 1 5 1 2 3 -i solution.txt -s
```
As a result, one gets two images (note that as there are many solutions, depending on the solver one might get a different solution here):

<p>
  <img alt="First unfolding" src="readme_imgs/Figure_1.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Second unfolding" src="readme_imgs/Figure_2.png" width="45%">
</p>
and a textual representation is left in the console:

```
*****2
****52
***352
****0*
****4*
***34*
**53**
*25***
*2****
*15***
43****
43****
4*****

*****3
****53
***253
****1*
****1*
***24*
**22**
*52***
*0****
*02***
34****
34****
3*****
```
We take the following convention for enumerating the faces:
```
0 -> Bottom face
1 -> Top face
2 -> Front face
3 -> Back face
4 -> Left face
5 -> Right face
```
<!-- 
### Decoder 
If sol.cnf is the v-lines output of the encoder file + first line saying satisfiable,
the solution can be decoded by:
```
python3 decoder.py -d a b c u v w -i sol.cnf 
```
where the option `-v 1` can be added to additionally show the edges preserved. -->

## Enumerating all solutions
To generate all sub-problems for enumerating unfoldings between boxes of dimensions $(a, b, c), (u, v, w)$ do:

```
sh enumerate.sh <a> <b> <c> <u> <v> <w>
```

For instance, `sh enumerate.sh 1 1 5 1 2 3` should finish about a minute. As a result, the folder `1x1x5_1x2x3_orient2_0` is created, containing 45 subfolders corresponding to 45 _pairs_. Now, run
```
cd 1x1x5_1x2x3_orient2_0/0_0_0_2_0_1_orient2_0
```
and observe both an `encoding.cnf` file, corresponding to the instance, and a `sol.txt` file with the satisfying assignments (3 for this pair, but other pairs may have more or 0). Running `cat sol.txt` now, we observe:

```
c
s SATISFIABLE
v 1 -2 3 4 -5 6 -7 8 -9 10 -11 -12 -13 -14 -15 -16 17 18 -19 -20 -21 -22 -23 -24 -25 -26 27 -28 -29 30 -31 -32 -33 34 35 36 37 -38 39 -40 -41 -42 -43 44 0
n Cummulative solutions 1
c
c
s SATISFIABLE
v 1 -2 -3 4 -5 6 -7 8 -9 10 -11 -12 -13 -14 -15 -16 17 18 -19 -20 -21 -22 -23 24 -25 -26 27 -28 -29 30 -31 -32 -33 34 35 36 37 -38 39 -40 -41 -42 -43 44 0
n Cummulative solutions 2
c
c
s SATISFIABLE
v 1 2 -3 4 -5 6 -7 8 -9 10 -11 -12 -13 -14 -15 -16 17 18 -19 -20 -21 -22 -23 -24 -25 -26 27 -28 -29 30 -31 -32 -33 34 35 36 37 -38 39 -40 -41 -42 -43 44 0
n Cummulative solutions 3
c
s 3 SOLUTIONS
FINISHED
```
We can divide this file into individuals solutions using `awk` :-)

```
awk '/^v /{file=sprintf("sol_%d.txt", ++count); print > file}' sol.txt
```
As a result, we have files `sol_1.txt`, `sol_2.txt`, and `sol_3.txt`.
These can now be decoded, as described above.

