/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__PARALLELIZATION
#define INQ__INTERFACE__PARALLELIZATION

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <systems/ions.hpp>

namespace inq {
namespace interface {

struct {

	constexpr auto name() const {
		return "parallelization";
	}

	constexpr auto one_line() const {
		return "Prints information about the parallelization of inq and the available options";
	}
	
	constexpr auto help() const {
		
		return R""""(

Parallelization of inq
====================== 

You can use multiple CPUs or multiple GPUs to make inq run faster and
simulate larger systems that what it could be possible with a single
processor. How to run inq in parallel depends on the specific system
you are running on. This will determine how many processors are used
by inq.

Inq has different ways to use those processors, that have to do with
the different ways of parallelizing the Kohn-Sham states. We now
describe the different parallelization approaches, in order of
granularity.

K-Point and Spin parallelization
--------------------------------

The first way of parallelizing a simulation is to divide the states
corresponding to each k-point among processors. So for example if you
have 12 k-points and 4 processors, you can give each process 3
k-points. This is an efficient method of parallelization since you
need to solve an independent eigenvalue problem for each k-point and
you only need communication for the calculation of the density.

This approach is usually limited by the total number of k-points that
might be not enough for all the processors that are available. Also
some parts of the calculation, like obtaining the Hartree and XC
potentials, are not k-point dependent and can become a bottleneck (due
to Amdhal's law). Finally, in the case of Hartree-Fock and hybrid
functionals there is an interaction between states of different
k-points are additional communication is needed.

Note than when running a (collinear) spin-polarized calculation there
are two independent set of eigenvalues for spin up and down. They
behave in the same way as different k-points, and can be used for
parallelization too. In inq k-point and spin parallelization is done
together, and it is usually referred as "k-point parallelization" for
simplicity.

State parallelization
---------------------

The second way of parallelizing the KS approach is to assign groups of
states to each processor. In this case the communication cost is
higher as orthogonalization routines, required for ground-state
calculations, need to communicate the states between processors.

For real-time TDDFT with semi-local functionals it is not necessary to
orthogonalize, so this is a very efficient parallelization scheme. It
is only limited by the number of avaible states.

Domain parallelization
----------------------

The last parallelization approach that inq uses is to split the
physical space in domains (both in real and Fourier spaces). Each
domain is then assigned to a processor. This usually requires the
largest amount of communication, especially for operations like
Fourier transforms.

Combined parallelization
------------------------

In order to scale to large numbers of processors (of around one GPU
per atom), inq can combine all of this parallelization
strategies. Effectively each processor will contain a few KS states
for a few k-points defined in small domain of the space.  Meaning that
the total number of processors we can use is the multiplication of the
processors we use for each parallelization strategy.

For example, if we divide the k-points among 8 processors, the states
among 4 processors and the domains among 2 processors we can use
8x4x2=64 processors.

What is the best way of dividing processors between the different
parallelization strategies is not always clear. It depends on the
specific computer where you are running, the type of calculation, the
size of system you are simulating, and the approximations (semi-local
functionals vs hybrids, for example).

By default inq will try to find the optimal division, but this is not
always easy. So the inq interface has some options to control how it
runs in parallel. It is recommended that you play with this options a
bit to find what works best, especially if you are going to run large
and/or long simulations.

Parallelization options
-----------------------

The three parallelization options are '-pk', '-ps' and '-pd', for
k-point, state and domain parallelization respectively. These are
options for the 'inq' executable and can be used with any command,
however it only make sense to pass them when using 'inq run' (in the
future some compute-intensive data analysis might depend on the proper
parallelization too). For example, to run the ground-state with 3
domains we can use

  `inq -pd 3 run ground state`

or if we want to run with 2 processor in the k-point parallelization
and 4 in the state parallelization

  `inq -pk 2 -ps 4 run ground state`

Passing a value of 1 means that that specific parallelization strategy
is disabled. The special value 'auto' means that inq can decide how
many processors to use in that strategy. The value 'auto' is the
default for all cases.

Note that there are some constraints to the values you can pass. The
first contraint is that any number of processors must de a divisor of
the total number of processors. Also the multiplication of the 3
values must be equal to the number of processors, if you leave some
options to the default 'auto', inq will fill them with the appropriate
values.

For the moment there is no option to control the parallelization from
python. We are working on this.

)"""";
	}
		
} const parallelization ;

}
}
#endif

#ifdef INQ_INTERFACE_PARALLELIZATION_UNIT_TEST
#undef INQ_INTERFACE_PARALLELIZATION_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
