# **Square_Skyrmion_Annealing**: Classical annealing Monte Carlo for Heisenberg model on square lattice

Copyright (C) 2017, Ya.V. Zhumagulov

## Required libraries

Square_Skyrmion_Annealing require TRIQS library
https://triqs.ipht.cnrs.fr/1.x/index.html

## Installation

```
git clone https://github.com/yaros72/Square-Skyrmion-Annealing.git ssa.src
mkdir ssa.build && cd ssa.build 
cmake -DTRIQS_PATH=path_to_triqs ../ssa.src
make
```
This create executable file ./Square_Skyrmion_Annealing in ssa.build directory which generate result as 'magnetization.dat' file.

Square_Skyrmion_Annealing.py is simple python wrapper (See Documentation).

## Documentation

For documentation and usage examples please see the hands on [jupyter notebook](Spiral-Phase-Example.ipynb)

## License

This application is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version (see <http://www.gnu.org/licenses/>).

It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
