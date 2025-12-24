#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear
inq cell  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0 scale 3.567095 A

inq ions insert fractional C 0.00 0.00 0.00
inq ions insert fractional C 0.25 0.25 0.25

inq electrons cutoff 35.0 Ha
inq electrons extra-states 3

inq kpoints gamma
inq kpoints insert -0.5 -0.5 -0.5 0.0

inq ground-state tolerance 1e-8

inq run ground state

inq util match `inq results ground-state energy total`           -10.949243966036  3e-5
inq util match `inq results ground-state energy kinetic`          11.423986855221  3e-5
inq util match `inq results ground-state energy eigenvalues`       0.796448318173  3e-5
inq util match `inq results ground-state energy hartree`           1.474429968732  3e-5
inq util match `inq results ground-state energy external`         -7.038773784808  3e-5
inq util match `inq results ground-state energy non-local`        -1.505554093563  3e-5
inq util match `inq results ground-state energy xc`               -4.568608783014  3e-5
inq util match `inq results ground-state energy nvxc`             -5.032070596140  3e-5
inq util match `inq results ground-state energy exact-exchange`    0.000000000000  3e-5
inq util match `inq results ground-state energy ion`             -10.734724128603  3e-5
