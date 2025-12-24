#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear

inq ions file `inq util test-data`/bn.poscar

inq electrons cutoff 35 hartree
inq electrons extra states 3

inq kpoints shifted grid 2 2 2

inq theory pbe

inq ground-state tolerance 1e-8

inq run ground-state

inq util match `inq results ground-state energy total`           -13.407436957736 3e-5
inq util match `inq results ground-state energy kinetic`           9.568700444445 3e-5
inq util match `inq results ground-state energy eigenvalues`      -0.947143661246 3e-5
inq util match `inq results ground-state energy hartree`           1.767696649837 3e-5
inq util match `inq results ground-state energy external`         -7.927178835552 3e-5
inq util match `inq results ground-state energy non-local`        -1.119663476805 3e-5
inq util match `inq results ground-state energy xc`               -4.406929320622 3e-5
inq util match `inq results ground-state energy nvxc`             -5.004395093009 3e-5
inq util match `inq results ground-state energy exact-exchange`    0.000000000000 3e-5
inq util match `inq results ground-state energy ion`             -11.290062419039 3e-5
