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
inq theory non-interacting
inq kpoints shifted grid 2 1 1
inq ground-state tolerance 1e-9

inq run ground-state

inq util match `inq results ground-state energy total`           -23.833852930943  3e-5
inq util match `inq results ground-state energy kinetic`          14.429591319530  3e-5
inq util match `inq results ground-state energy eigenvalues`       7.649767564156  3e-5
inq util match `inq results ground-state energy hartree`           0.000000000000  3e-5
inq util match `inq results ground-state energy external`        -12.018531748189  3e-5
inq util match `inq results ground-state energy non-local`         5.238707992815  3e-5
inq util match `inq results ground-state energy xc`                0.000000000000  3e-5
inq util match `inq results ground-state energy nvxc`              0.000000000000  3e-5
inq util match `inq results ground-state energy exact-exchange`    0.000000000000  3e-5
inq util match `inq results ground-state energy ion`             -31.483620495100  3e-5

