/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__ENERGY
#define INQ__HAMILTONIAN__ENERGY

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <operations/shift.hpp>

#include <tinyformat/tinyformat.h>

namespace inq {
namespace hamiltonian {

	class energy {

		double ion_ = 0.0;
		double ion_kinetic_ = 0.0;		
		double kinetic_     = 0.0;
		double eigenvalues_ = 0.0;
		double external_ = 0.0;
		double non_local_ = 0.0;
		double hartree_ = 0.0;
		double xc_ = 0.0;
		double nvxc_ = 0.0;
		double exact_exchange_ = 0.0;
		double zeeman_ener_ = 0.0;

#ifdef ENABLE_CUDA
public:
#endif
		
		template <typename OccType, typename ArrayType>
		static double occ_sum(OccType const & occupations, ArrayType const & array) {
			CALI_CXX_MARK_FUNCTION;
			
			assert(occupations.size() == array.size());
			return gpu::run(gpu::reduce(array.size()), 0.0, [occ = begin(occupations), arr = begin(array)] GPU_LAMBDA (auto ip) {
				return occ[ip]*real(arr[ip]);
			});
		}

		template <typename Occupations, typename Array, typename Norms>
		static double occ_sum(Occupations const & occupations, Array const & array, Norms const & norms) {
			CALI_CXX_MARK_FUNCTION;
			
			assert(occupations.size() == array.size());
			return gpu::run(gpu::reduce(array.size()), 0.0, [occ = begin(occupations), arr = begin(array), nor = begin(norms)] GPU_LAMBDA (auto ip) {
				return occ[ip]*real(arr[ip])/real(nor[ip]);
			});
		}

public:
		
		energy() = default;

		template <typename HamType, typename ElType>
		auto calculate(HamType const & ham, ElType & el) {

			CALI_CXX_MARK_SCOPE("energy::calculate");

			auto normres = gpu::array<complex, 2>({static_cast<gpu::array<complex, 2>::size_type>(el.kpin().size()), el.max_local_spinor_set_size()});

			kinetic_ = 0.0;
			eigenvalues_ = 0.0;
			non_local_ = 0.0;
			exact_exchange_ = 0.0;
			
			int iphi = 0;
			for(auto & phi : el.kpin()){

				auto kin_ev = ham.kinetic_expectation_value(phi);
				auto norms = operations::overlap_diagonal(phi);

				kinetic_ += occ_sum(el.occupations()[iphi], kin_ev, norms);
				
				{
					CALI_CXX_MARK_SCOPE("energy::calculate::eigenvalues");
					auto residual = ham(phi);
					el.eigenvalues()[iphi] = operations::overlap_diagonal_normalized(residual, phi, operations::real_part{});
					operations::shift(-1.0, el.eigenvalues()[iphi], phi, residual);
					normres[iphi] = operations::overlap_diagonal(residual);
					eigenvalues_ += occ_sum(el.occupations()[iphi], el.eigenvalues()[iphi]);
				}

				non_local_ += ham.non_local_energy(phi, el.occupations()[iphi], /*reduce_states = */ false);
				
				if(ham.exchange().enabled()){
					CALI_CXX_MARK_SCOPE("energy::calculate::exchange");
					auto exchange_me = operations::overlap_diagonal_normalized(ham.exchange()(phi), phi);
					exact_exchange_ += 0.5*occ_sum(el.occupations()[iphi], exchange_me);
				}

				iphi++;
			}

			if(el.kpin_states_comm().size() > 1){	
				CALI_CXX_MARK_SCOPE("energy::calculate::reduce");

				double red[4] = {kinetic_, eigenvalues_, non_local_, exact_exchange_};
				el.kpin_states_comm().all_reduce_in_place_n(red, 4);
				kinetic_         = red[0];
				eigenvalues_     = red[1];
				non_local_       = red[2];
				exact_exchange_  = red[3];
			}

			return normres;
		}
	
		auto kinetic() const {
			return kinetic_;
		}

		void kinetic(double const & val) {
			kinetic_ = val;
		}
		
		auto total() const {
			return kinetic() + hartree_ + external_ + non_local_ + xc_ + exact_exchange_ + ion_ + ion_kinetic_;
		}

		auto & eigenvalues() const {
			return eigenvalues_;
		}
		
		void eigenvalues(double const & val) {
			eigenvalues_ = val;
		}

		auto & hartree() const {
			return hartree_;
		}

		void hartree(double const & val) {
			hartree_ = val;
		}

		auto & external() const {
			return external_;
		}

		void external(double const & val) {
			external_ = val;
		}

		auto & non_local() const {
			return non_local_;
		}

		void non_local(double const & val) {
			non_local_ = val;
		}

		auto & xc() const {
			return xc_;
		}

		void xc(double const & val) {
			xc_ = val;
		}

		auto & nvxc() const {
			return nvxc_;
		}

		void nvxc(double const & val) {
			nvxc_ = val;
		}

		auto & zeeman_energy() const {
			return zeeman_ener_;
		}

		void zeeman_energy(double const & val) {
			zeeman_ener_ = val;
		}

		auto & exact_exchange() const {
			return exact_exchange_;
		}

		void exact_exchange(double const & val) {
			exact_exchange_ = val;
		}

		auto & ion() const {
			return ion_;
		}

		void ion(double const & val) {
			ion_ = val;
		}

		auto & ion_kinetic() const {
			return ion_kinetic_;
		}

		void ion_kinetic(double const & val) {
			ion_kinetic_ = val;
		}

		void save(parallel::communicator & comm, std::string const & dirname) const {
			auto error_message = "INQ error: Cannot save the energy to directory '" + dirname + "'.";

			utils::create_directory(comm, dirname);
			utils::save_value(comm, dirname + "/ion",            ion_,         error_message);
			utils::save_value(comm, dirname + "/ion_kinetic",    ion_kinetic_, error_message);
			utils::save_value(comm, dirname + "/kinetic",        kinetic_,        error_message);
			utils::save_value(comm, dirname + "/eigenvalues",    eigenvalues_, error_message);
			utils::save_value(comm, dirname + "/external",       external_,    error_message);
			utils::save_value(comm, dirname + "/non-local",      non_local_,      error_message);
			utils::save_value(comm, dirname + "/hartree",        hartree_,     error_message);
			utils::save_value(comm, dirname + "/xc",             xc_,          error_message);
			utils::save_value(comm, dirname + "/nvxc",           nvxc_,        error_message);
			utils::save_value(comm, dirname + "/exact_exchange", exact_exchange_, error_message);
			utils::save_value(comm, dirname + "/zeeman_energy",  zeeman_ener_,    error_message);
		}

		static auto load(std::string const & dirname) {
			auto error_message = "INQ error: Cannot load the energy from directory '" + dirname + "'.";
			energy en;

			utils::load_value(dirname + "/ion",             en.ion_,            error_message);
			utils::load_value(dirname + "/ion_kinetic",     en.ion_kinetic_,    error_message);
			utils::load_value(dirname + "/kinetic",         en.kinetic_,        error_message);
			utils::load_value(dirname + "/eigenvalues",     en.eigenvalues_,    error_message);
			utils::load_value(dirname + "/external",        en.external_,       error_message);
			utils::load_value(dirname + "/non-local",       en.non_local_,      error_message);
			utils::load_value(dirname + "/hartree",         en.hartree_,        error_message);
			utils::load_value(dirname + "/xc",              en.xc_,             error_message);
			utils::load_value(dirname + "/nvxc",            en.nvxc_,           error_message);
			utils::load_value(dirname + "/exact_exchange",  en.exact_exchange_, error_message);
			utils::load_value(dirname + "/zeeman_energy",   en.zeeman_ener_,    error_message);
			
			return en;
		}
		
		template<class OStream>
		friend OStream & operator<<(OStream & out, energy const & self){

			tfm::format(out, "Energy:\n");
			tfm::format(out, "  total          = %20.12f Ha\n", self.total());
			tfm::format(out, "  kinetic        = %20.12f Ha\n", self.kinetic());
			tfm::format(out, "  eigenvalues    = %20.12f Ha\n", self.eigenvalues_);
			tfm::format(out, "  hartree        = %20.12f Ha\n", self.hartree());
			tfm::format(out, "  external       = %20.12f Ha\n", self.external());
			tfm::format(out, "  non-local      = %20.12f Ha\n", self.non_local());
			tfm::format(out, "  xc             = %20.12f Ha\n", self.xc());
			tfm::format(out, "  nvxc           = %20.12f Ha\n", self.nvxc());
			tfm::format(out, "  exact-exchange = %20.12f Ha\n", self.exact_exchange());
			tfm::format(out, "  ion            = %20.12f Ha\n", self.ion());
			tfm::format(out, "  zeeman-energy  = %20.12f Ha\n", self.zeeman_energy());
			tfm::format(out, "\n");

			return out;
		}
		
	};

}
}
#endif

#ifdef INQ_HAMILTONIAN_ENERGY_UNIT_TEST
#undef INQ_HAMILTONIAN_ENERGY_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace Catch::literals;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	hamiltonian::energy en;

	en.ion(1.0);
	en.ion_kinetic(2.0);
	en.kinetic(2.5);
	en.eigenvalues(3.0);
	en.external(4.0);
	en.non_local(5.0);
	en.hartree(6.0);
	en.xc(7.0);
	en.nvxc(8.0);
	en.exact_exchange(10.0);

	CHECK(en.ion() == 1.0);
	CHECK(en.ion_kinetic() == 2.0);
	CHECK(en.kinetic()     == 2.5);
	CHECK(en.eigenvalues() == 3.0);
	CHECK(en.external() == 4.0);
	CHECK(en.non_local() == 5.0);
	CHECK(en.hartree() == 6.0);
	CHECK(en.xc() == 7.0);
	CHECK(en.nvxc() == 8.0);
	CHECK(en.exact_exchange() == 10.0);
	
	en.save(comm, "save_energy");
	auto read_en = hamiltonian::energy::load("save_energy");
	
	CHECK(read_en.ion() == 1.0);
	CHECK(read_en.ion_kinetic() == 2.0);
	CHECK(read_en.kinetic()     == 2.5);
	CHECK(read_en.eigenvalues() == 3.0);
	CHECK(read_en.external() == 4.0);
	CHECK(read_en.non_local() == 5.0);
	CHECK(read_en.hartree() == 6.0);
	CHECK(read_en.xc() == 7.0);
	CHECK(read_en.nvxc() == 8.0);
	CHECK(read_en.exact_exchange() == 10.0);
	
}
#endif

