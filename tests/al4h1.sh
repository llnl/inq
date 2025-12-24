#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear
inq cell cubic 7.6524459 bohr

inq ions insert fractional Al  0.0  0.0  0.0
inq ions insert fractional Al  0.0  0.5  0.5
inq ions insert fractional Al  0.5  0.0  0.5 
inq ions insert fractional Al  0.5  0.5  0.0
inq ions insert fractional H   0.1  0.2  0.3 

inq electrons cutoff 30 hartree
inq electrons extra states 1
inq electrons temperature 300 Kelvin
inq theory pbe
inq kpoints shifted grid 2 2 2
inq ground-state tolerance 1e-8

inq run ground-state

inq util match `inq results ground-state energy total`                -9.802823539831 3e-5
inq util match `inq results ground-state energy kinetic`               4.201409498644 3e-5
inq util match `inq results ground-state energy eigenvalues`           0.602228265392 3e-5
inq util match `inq results ground-state energy hartree`               0.219494927949 3e-5
inq util match `inq results ground-state energy external`             -0.564088418335 3e-5
inq util match `inq results ground-state energy non-local`             1.426927243301 3e-5
inq util match `inq results ground-state energy xc`                   -4.768194678158 3e-5
inq util match `inq results ground-state energy nvxc`                 -4.901009914115 3e-5
inq util match `inq results ground-state energy exact-exchange`        0.000000000000 3e-5
inq util match `inq results ground-state energy ion`                 -10.318372113231 3e-5

inq util match `inq results ground-state forces 0`                  -2.09566529710921191365e-02	-3.85828527747746410914e-02	-4.91494535948646393830e-02  3e-5
inq util match `inq results ground-state forces 1`                  -2.09566351829116766237e-02  4.91493373873537267582e-02	 3.85829597069514557139e-02  3e-5
inq util match `inq results ground-state forces 2`                   6.65782414508742353271e-03	-1.28182788049416677978e-02	 1.28184191878267378373e-02  3e-5
inq util match `inq results ground-state forces 3`                   7.67568625007923437120e-03	 1.25051202110598655426e-02	-1.25052564658440050482e-02  3e-5
inq util match `inq results ground-state forces 4`                   2.75725325145750391198e-02	-1.02323287043417452541e-02	 1.02323623136605021400e-02  3e-5

inq electrons extra states 0
inq real-time num-steps 30
inq real-time time-step 0.055 atu

inq run real-time

inq util match `inq results real-time total-steps`                 30    1e-16
inq util match `inq results real-time total-time`                   1.65 1e-16
inq util match `inq results real-time total-energy  0`             -9.80282353983139032039e+00 3e-5
inq util match `inq results real-time total-energy 10`             -9.80282353983139564946e+00 3e-5
inq util match `inq results real-time total-energy 20`             -9.80282353983139032039e+00 3e-5
inq util match `inq results real-time total-energy 30`             -9.80282353983124821184e+00 3e-5
