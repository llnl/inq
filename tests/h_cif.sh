#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear

inq ions file `inq util test-data`/H.cif

inq electrons cutoff 30 hartree
inq theory pbe
inq ground-state tolerance 1e-8
inq ground-state mixing 0.1

inq run ground-state

inq util match `inq results ground-state energy total`           -2.356866852184 3e-5
inq util match `inq results ground-state energy kinetic`          2.132039018586 3e-5
inq util match `inq results ground-state energy eigenvalues`     -1.510974587511 3e-5
inq util match `inq results ground-state energy hartree`          0.905608655081 3e-5
inq util match `inq results ground-state energy external`        -3.325211379435 3e-5
inq util match `inq results ground-state energy non-local`       -0.393268769855 3e-5
inq util match `inq results ground-state energy xc`              -1.352340477011 3e-5
inq util match `inq results ground-state energy nvxc`            -1.735750766969 3e-5
inq util match `inq results ground-state energy exact-exchange`   0.000000000000 3e-5
inq util match `inq results ground-state energy ion`             -0.323693899550 3e-5

#TODO: check the eigenvalues to be -0.401182208666 and -0.354512952159
