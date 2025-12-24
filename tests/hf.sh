#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear

inq cell cubic 10.0 bohr finite

dist=0.917

inq species add Hloc element H
inq species Hloc file `inq util test-data`/H.blyp-vbc.UPF

inq ions insert Hloc 0.0 -$dist/2 0.0 angstrom
inq ions insert F    0.0  $dist/2 0.0 angstrom

inq electrons cutoff 30 hartree
inq ground-state tolerance 1e-8

inq run gs

inq util match `inq results ground-state energy total`           -25.307080856594 3e-5
inq util match `inq results ground-state energy kinetic`          16.642307017977 3e-5
inq util match `inq results ground-state energy eigenvalues`      -4.626023751667 3e-5
inq util match `inq results ground-state energy hartree`          25.434531554199 3e-5
inq util match `inq results ground-state energy external`        -62.899445254320 3e-5
inq util match `inq results ground-state energy non-local`        -2.648488639381 3e-5
inq util match `inq results ground-state energy xc`               -5.875506228977 3e-5
inq util match `inq results ground-state energy nvxc`             -6.589459984342 3e-5
inq util match `inq results ground-state energy exact-exchange`    0.000000000000 3e-5
inq util match `inq results ground-state energy ion`               4.039520693908 3e-5

