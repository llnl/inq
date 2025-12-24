#!/bin/sh

set -e #make the script fail if a command fails
set -x #output commands to the terminal

inq clear
inq cell  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0 scale 3.567095 A

inq ions insert fractional C 0.00 0.00 0.00
inq ions insert fractional C 0.25 0.25 0.25

inq electrons cutoff 35.0 Ha
inq electrons extra-states 3

inq kpoints shifted grid 2 2 2

inq ground-state tolerance 1e-8
inq ground-state max-steps 400

inq theory PBE0

inq run ground state

inq util match `inq results ground-state energy total`           -11.570967417742  3e-5
inq util match `inq results ground-state energy kinetic`           8.397347945483  3e-5
inq util match `inq results ground-state energy eigenvalues`      -0.751045499406  3e-5
inq util match `inq results ground-state energy hartree`           0.939543496422  3e-5
inq util match `inq results ground-state energy external`         -5.739320858820  3e-5
inq util match `inq results ground-state energy non-local`        -0.573240698707  3e-5
inq util match `inq results ground-state energy xc`               -3.345415741685  3e-5
inq util match `inq results ground-state energy nvxc`             -3.684604016542  3e-5
inq util match `inq results ground-state energy exact-exchange`   -0.515157431832  3e-5
inq util match `inq results ground-state energy ion`             -10.734724128603  3e-5

inq theory Hartree-Fock

inq run ground state

inq util match `inq results ground-state energy total`            -9.787830957944  3e-5
inq util match `inq results ground-state energy kinetic`           8.159350915885  3e-5
inq util match `inq results ground-state energy eigenvalues`      -0.248787497148  3e-5
inq util match `inq results ground-state energy hartree`           0.880208047166  3e-5
inq util match `inq results ground-state energy external`         -5.500775282964  3e-5
inq util match `inq results ground-state energy non-local`        -0.516001794456  3e-5
inq util match `inq results ground-state energy xc`                0.000000000000  3e-5
inq util match `inq results ground-state energy nvxc`              0.000000000000  3e-5
inq util match `inq results ground-state energy exact-exchange`   -2.075888714972  3e-5
inq util match `inq results ground-state energy ion`             -10.734724128603  3e-5
