/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__GROUND_STATE__EIGENVALUE_OUTPUT
#define INQ__GROUND_STATE__EIGENVALUE_OUTPUT

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cfloat>
#include <systems/electrons.hpp>

#include <spdlog/spdlog.h>
#include <utils/profiling.hpp>

namespace inq {
namespace ground_state {

template <class>
class full_eigenvalues_output;

class eigenvalues_output {

protected:
	
	int nspin_;
	int nkpoints_;
	gpu::array<int, 1> all_kpoint_index;
	gpu::array<int, 1> all_spin_index;
	gpu::array<int, 1> all_states_index;
	gpu::array<double, 1> all_eigenvalues;
	gpu::array<double, 1> all_occupations;
	gpu::array<complex, 1> all_normres;
	systems::electrons const & electrons_;
	
public:
	
	template <typename NormResType>
	eigenvalues_output(systems::electrons const & el, NormResType const & normres):
		nspin_(el.states().num_spin_indices()),
		nkpoints_(el.brillouin_zone().size()),
		electrons_(el)
	{
		gpu::array<int, 2> kpoint_index({el.kpin_part().local_size(), el.max_local_spinor_set_size()});
		gpu::array<int, 2> spin_index({el.kpin_part().local_size(), el.max_local_spinor_set_size()});
		gpu::array<int, 2> state_index({el.kpin_part().local_size(), el.max_local_spinor_set_size()});
		gpu::array<double, 2> occs({el.kpin_part().local_size(), el.max_local_spinor_set_size()});
		
		auto iphi = 0;
		for(auto & phi : el.kpin()){
			auto ik = el.kpoint_index(phi);
			for(int ist = 0; ist < el.max_local_spinor_set_size(); ist++){
				kpoint_index[iphi][ist] = ik;
				spin_index[iphi][ist] = phi.spin_index();
				state_index[iphi][ist] = phi.set_part().local_to_global(ist).value();
			occs[iphi][ist] = 0.0;
			if(fabs(el.kpin_weights()[iphi]) > 1e-14) occs[iphi][ist] = el.occupations()[iphi][ist]/el.kpin_weights()[iphi];
			}
			iphi++;
		}

		auto part = parallel::arbitrary_partition(el.kpin_part().local_size()*el.max_local_spinor_set_size(), el.kpin_states_comm());
		
		all_kpoint_index = parallel::gather(+kpoint_index.flatted(), part, el.kpin_states_comm(), 0);
		if(el.kpin_states_comm().root()){
			for([[maybe_unused]] auto index : all_kpoint_index) assert(index >= 0 and index < electrons_.brillouin_zone().size());
		}
		
		all_spin_index = parallel::gather(+spin_index.flatted(), part, el.kpin_states_comm(), 0);
		all_states_index = parallel::gather(+state_index.flatted(), part, el.kpin_states_comm(), 0);
		all_eigenvalues = parallel::gather(+el.eigenvalues().flatted(), part, el.kpin_states_comm(), 0);
		all_occupations = parallel::gather(+occs.flatted(), part, el.kpin_states_comm(), 0);
		all_normres = parallel::gather(+normres.flatted(), part, el.kpin_states_comm(), 0);
	}

	static std::string spin_string(int index){
		if(index == 0) return "\u25B2";
		return "\u25BC";
	}

	template <class OStream>
	friend OStream& operator<<(OStream & out, eigenvalues_output const & self){

		auto order = gpu::array<int, 1>(self.all_eigenvalues().size());
		
		for(int iorder = 0; iorder < order.size(); iorder++) order[iorder] = iorder;
		
		std::sort(order.begin(), order.end(), [evs = self.all_eigenvalues](auto io, auto jo){ return evs[io] < evs[jo]; });
		
		auto const print_range = 8;
		
		//define the LUMO as the first state with ocupation below 0.1 (this is only for output purposes)
		auto lumo_index = order.size() - 1;
		for(int iorder = 0; iorder < order.size(); iorder++) {
			if(self.all_occupations[order[iorder]] >= 0.1) continue;
			lumo_index = iorder;
			break;
		}
		
		int skipped = 0;
		double minres = 1000.0;
		double maxres = 0.0;
		for(int iorder = 0; iorder < order.size(); iorder++) {
			auto ieig = order[iorder];
			auto print = (iorder < print_range) or (abs(iorder - lumo_index) < print_range) or (order.size() - 1 - iorder < print_range);
			
			if(not print) {
				skipped++;
				minres = std::min(minres, fabs(self.all_normres[ieig]));
				maxres = std::max(minres, fabs(self.all_normres[ieig]));
				continue;
			}
			
			if(skipped > 0) {
				fmt::print(out, "    [output of {:d} eigenvalues suppressed,  minres = {:5.0e}  maxres = {:5.0e}]\n", skipped, minres, maxres);
				skipped = 0;
				minres = 1000.0;
				maxres = 0.0;			
			}

			auto kpoint = self.electrons_.brillouin_zone().kpoint(self.all_kpoint_index[ieig])/(2.0*M_PI);
			
			if(self.nkpoints_ > 1) fmt::print(out, "  kpt = ({:5.2f},{:5.2f},{:5.2f})", kpoint[0], kpoint[1], kpoint[2]);
			if(self.nspin_    > 1) fmt::print(out, "  spin = {}", spin_string(self.all_spin_index[ieig]));
			fmt::print(out, "  st = {:4d}  occ = {:4.3f}  evalue = {:18.12f}  res = {:5.0e}\n",
									self.all_states_index[ieig] + 1, self.all_occupations[ieig], real(self.all_eigenvalues[ieig]), fabs(self.all_normres[ieig]));
		}

		return out;
	}

	template <class Base>
	class full_ : public Base {

	public:
		
		full_(Base bas):
			Base(std::move(bas)){
		}
		
		template <class OStream>
		friend OStream& operator<<(OStream & out, full_ const & self){
			auto curkpoint = -1;
			auto curspin = -1;
			
			for(int ieig = 0; ieig < self.all_eigenvalues().size(); ieig++) {

				if(curkpoint != self.all_kpoint_index[ieig] or curspin != self.all_spin_index[ieig]){

					curkpoint = self.all_kpoint_index[ieig];
					curspin = self.all_spin_index[ieig];

					auto kpoint = self.electrons_.brillouin_zone().kpoint(self.all_kpoint_index[ieig])/(2.0*M_PI);

					if(self.nkpoints_ > 1) fmt::print(out, "  kpt = (:5.2f},{:5.2f},{:5.2f})", kpoint[0], kpoint[1], kpoint[2]);
					if(self.nspin_    > 1) fmt::print(out, "  spin = {}", spin_string(self.all_spin_index[ieig]));
					if(self.nkpoints_ > 1 or self.nspin_    > 1) fmt::print(out, "\n");
				}
				
				fmt::print(out, "    st = {:4d}  occ = {:4.3f}  evalue = {:18.12f} Ha ({:18.12f} eV)  res = {:5.0e}\n",
										self.all_states_index[ieig] + 1, self.all_occupations[ieig], real(self.all_eigenvalues[ieig]), real(self.all_eigenvalues[ieig])*27.211383, fabs(self.all_normres[ieig]));
			}
			
			return out;
		}
		
	};
	
	auto full() const {
		return full_<eigenvalues_output>(*this);
	}
	
};

}
}
#endif

#ifdef INQ_GROUND_STATE_EIGENVALUE_OUTPUT_UNIT_TEST
#undef INQ_GROUND_STATE_EIGENVALUE_OUTPUT_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif

