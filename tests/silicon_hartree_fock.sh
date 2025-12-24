#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear
inq cell cubic 10.18 bohr

inq ions insert fractional Si  0.0  0.0  0.0 
inq ions insert fractional Si  0.25 0.25 0.25
inq ions insert fractional Si  0.5  0.5  0.0 
inq ions insert fractional Si  0.75 0.75 0.25
inq ions insert fractional Si  0.5  0.0  0.5 
inq ions insert fractional Si  0.75 0.25 0.75
inq ions insert fractional Si  0.0  0.5  0.5 
inq ions insert fractional Si  0.25 0.75 0.75

inq electrons cutoff 25 hartree
inq electrons extra-states 4

inq ground-state tolerance 1e-4
inq run ground-state

inq theory Hartree-Fock
inq ground-state tolerance 1e-8
inq run ground-state

inq util match `inq results ground-state energy total`         -30.503525726628  3e-5
inq util match `inq results ground-state energy kinetic`        13.264994256161  3e-5
inq util match `inq results ground-state energy eigenvalues`    -6.183870496831  3e-5
inq util match `inq results ground-state energy hartree`         2.508104348642  3e-5
inq util match `inq results ground-state energy external`       -9.272300998267  3e-5
inq util match `inq results ground-state energy non-local`       4.151366775879  3e-5
inq util match `inq results ground-state energy xc`              0.000000000000  3e-5
inq util match `inq results ground-state energy nvxc`            0.000000000000  3e-5
inq util match `inq results ground-state energy exact-exchange` -9.672069613944  3e-5
inq util match `inq results ground-state energy ion`           -31.483620495100  3e-5
