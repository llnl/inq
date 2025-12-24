#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear

inq cell cubic 15.0 bohr finite

inq species add Hloc element H
inq species Hloc file `inq util test-data`/H.blyp-vbc.UPF
inq ions insert Hloc 150.0 -30.0 0.0 bohr

inq electrons cutoff 40 hartree
inq theory non interacting
inq ground-state tolerance 1e-8
inq ground-state max-steps 400

inq run gs

inq util match `inq results ground-state energy total`           -0.499243892116 3e-5
inq util match `inq results ground-state energy kinetic`          0.489074177507 3e-5
inq util match `inq results ground-state energy eigenvalues`     -0.499243892116 3e-5
inq util match `inq results ground-state energy hartree`          0.000000000000 3e-5
inq util match `inq results ground-state energy external`        -0.988318069623 3e-5
inq util match `inq results ground-state energy non-local`        0.000000000000 3e-5
inq util match `inq results ground-state energy xc`               0.000000000000 3e-5
inq util match `inq results ground-state energy nvxc`             0.000000000000 3e-5
inq util match `inq results ground-state energy exact-exchange`   0.000000000000 3e-5
inq util match `inq results ground-state energy ion`              0.000000000000 3e-5

inq theory lda
inq run gs

inq util match `inq results ground-state energy total`           -0.445349621508 3e-5
inq util match `inq results ground-state energy kinetic`          0.414993735403 3e-5
inq util match `inq results ground-state energy eigenvalues`     -0.234117323185 3e-5
inq util match `inq results ground-state energy hartree`          0.281443638065 3e-5
inq util match `inq results ground-state energy external`        -0.910162044475 3e-5
inq util match `inq results ground-state energy non-local`        0.000000000000 3e-5
inq util match `inq results ground-state energy xc`              -0.231624950501 3e-5
inq util match `inq results ground-state energy nvxc`            -0.301836290243 3e-5
inq util match `inq results ground-state energy exact-exchange`   0.000000000000 3e-5
inq util match `inq results ground-state energy ion`              0.000000000000 3e-5
