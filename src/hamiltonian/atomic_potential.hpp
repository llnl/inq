/* -*- indent-tabs-mode: t -*- */

#ifndef ATOMIC_POTENTIAL_HPP
#define ATOMIC_POTENTIAL_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include <utils/raw_pointer_cast.hpp>
#include <config/path.hpp>

#include <pseudopod/set.hpp>
#include <pseudopod/pseudopotential.hpp>

#include <basis/spherical_grid.hpp>
#include <basis/double_grid.hpp>
#include <gpu/array.hpp>
#include <operations/integral.hpp>
#include <options/electrons.hpp>
#include <parallel/partition.hpp>
#include <solvers/poisson.hpp>
#include <states/ks_states.hpp>
#include <observables/magnetization.hpp>


#include <unordered_map>

#include <mpi3/environment.hpp>

#include <utils/profiling.hpp>

namespace inq {
namespace hamiltonian {

	class atomic_potential {

	public:
		
		using pseudopotential_type = pseudo::pseudopotential<gpu::array<double, 1>>;

	private:
		
		pseudo::math::erf_range_separation sep_;
		pseudo::set default_pseudo_set_;
		std::unordered_map<std::string, pseudopotential_type> pseudopotential_list_;
		bool has_nlcc_;
		bool has_overlap_;
		basis::double_grid double_grid_;

	public:

		template <class SpeciesList>
		atomic_potential(SpeciesList const & species_list, double gcutoff, options::electrons const & conf = {}):
			sep_(0.625), //this is the default from octopus
			default_pseudo_set_(species_list.pseudopotentials()),
			double_grid_(conf.double_grid_value())
		{

			CALI_CXX_MARK_FUNCTION;

			gcutoff *= double_grid_.spacing_factor(); 
			
			has_nlcc_ = false;
			has_overlap_= false;

			for(auto const & species : species_list) {
				if(pseudopotential_list_.find(species.symbol()) != pseudopotential_list_.end()) throw std::runtime_error("INQ Error: duplicated species");
				
				auto file_path = std::string{};
				if(species.has_file()) {
					file_path = species.file_path();
				} else if(species.has_pseudo_set()) {
					auto set = pseudo::set{species.pseudo_set()};
					if(!set.has(species)) throw std::runtime_error("inq error: pseudopotential for element " + species.symbol() + " not found in requested set.");
					file_path = set.file_path(species);
				} else {
					if(!default_pseudo_set_.has(species)) throw std::runtime_error("inq error: pseudopotential for element " + species.symbol() + " not found.");
					file_path = default_pseudo_set_.file_path(species);
				}
				assert(not file_path.empty());

				//sorry for this, emplace has a super ugly syntax
				auto insert = pseudopotential_list_.emplace(std::piecewise_construct, std::make_tuple(species.symbol()), std::make_tuple(file_path, sep_, gcutoff, species.filter_pseudo()));
				auto & pseudo = insert.first->second;
				
				has_nlcc_ = has_nlcc_ or pseudo.has_nlcc_density();
				has_overlap_ = has_overlap_ or pseudo.has_overlap();
			}

		}
		
		int num_species() const {
			return pseudopotential_list_.size();
		}

		template <class SymbolsList>
		auto num_electrons(SymbolsList const symbols_list) {
			double nel = 0.0;
			for(auto symbol : symbols_list) {
				auto map_ref = pseudopotential_list_.find(symbol);
				assert(map_ref != pseudopotential_list_.end());
				auto & pseudo = map_ref->second;
				nel += pseudo.valence_charge();
			}
			return nel;
		}

		auto & pseudo_for_element(ionic::species const & el) const {
			return pseudopotential_list_.at(el.symbol());
		}

		template <class CommType, class basis_type, class ions_type>
		basis::field<basis_type, double> local_potential(CommType & comm, const basis_type & basis, const ions_type & ions, int single_atom = -1) const {

			CALI_CXX_MARK_SCOPE("atomic_potential::local_potential");
			
			basis::field<basis_type, double> potential(basis);

			parallel::partition part(ions.size(), comm);
			
			potential.fill(0.0);
			
			for(auto iatom = part.start(); iatom < part.end(); iatom++){

				if(single_atom >= 0 and single_atom != iatom) continue;
				
				auto atom_position = ions.positions()[iatom];
				
				auto & ps = pseudo_for_element(ions.species(iatom));
				basis::spherical_grid sphere(basis, atom_position, ps.short_range_potential_radius());

				if(not double_grid_.enabled()){

					gpu::run(sphere.size(),
									 [pot = begin(potential.cubic()),
										sph = sphere.ref(),
										spline = ps.short_range_potential().function()] GPU_LAMBDA (auto ipoint){
										 auto rr = sph.distance(ipoint);
										 auto potential_val = spline(rr);
										 gpu::atomic(pot[sph.grid_point(ipoint)[0]][sph.grid_point(ipoint)[1]][sph.grid_point(ipoint)[2]]) += potential_val;
									 });

				} else {

					CALI_CXX_MARK_SCOPE("atomic_potential::double_grid");
					
					gpu::run(sphere.size(),
									 [pot = begin(potential.cubic()),
										sph = sphere.ref(),
										spline = ps.short_range_potential().function(),
										dg = double_grid_.ref(),
										spac = basis.rspacing(), cell = basis.cell()] GPU_LAMBDA (auto ipoint){
										 gpu::atomic(pot[sph.grid_point(ipoint)[0]][sph.grid_point(ipoint)[1]][sph.grid_point(ipoint)[2]])
											 +=	dg.value([spline] GPU_LAMBDA (auto pos) { return spline(pos.length()); }, spac, cell.to_cartesian(sph.point_pos(ipoint)));
									 });
				}
			}
			
			potential.all_reduce(comm);
			return potential;			
		}

		////////////////////////////////////////////////////////////////////////////////////
		
		template <class CommType, class basis_type, class ions_type>
		basis::field<basis_type, double> ionic_density(CommType & comm, const basis_type & basis, const ions_type & ions, int single_atom = -1) const {

			CALI_CXX_MARK_FUNCTION;

			parallel::partition part(ions.size(), comm);
			
			basis::field<basis_type, double> density(basis);
			
			density.fill(0.0);

			for(auto iatom = part.start(); iatom < part.end(); iatom++){

				if(single_atom >= 0 and single_atom != iatom) continue;
				
				auto atom_position = ions.positions()[iatom];
				
				auto & ps = pseudo_for_element(ions.species(iatom));
				basis::spherical_grid sphere(basis, atom_position, sep_.long_range_density_radius());

				//OPTIMIZATION: this should be done in parallel for atoms too
				gpu::run(sphere.size(),
								 [dns = begin(density.cubic()),
									sph = sphere.ref(),
									chrg = ps.valence_charge(),
									sp = sep_] GPU_LAMBDA (auto ipoint){
									 double rr = sph.distance(ipoint);
									 gpu::atomic(dns[sph.grid_point(ipoint)[0]][sph.grid_point(ipoint)[1]][sph.grid_point(ipoint)[2]]) += chrg*sp.long_range_density(rr);
								 });
			}

			density.all_reduce(comm);
			return density;			
		}

		////////////////////////////////////////////////////////////////////////////////////
		
		template <class CommType, class basis_type, class ions_type>
		basis::field_set<basis_type, double> atomic_electronic_density(CommType & comm, const basis_type & basis, const ions_type & ions, states::ks_states const & states, std::vector<vector3<double>> const & magnet_dir = {}) const {

			CALI_CXX_MARK_FUNCTION;

			parallel::partition part(ions.size(), comm);

			auto nspin = states.num_density_components();
			basis::field_set<basis_type, double> density(basis, nspin);

			density.fill(0.0);
			
			if (magnet_dir.size() > 0) {
				assert(magnet_dir.size() == ions.size());
			}
			
			for(auto iatom = part.start(); iatom < part.end(); iatom++){
				
				auto atom_position = ions.positions()[iatom];
				
				vector3<double> atomic_magnet = {0.0, 0.0, 0.0};
				if (magnet_dir.size() > 0) {
					atomic_magnet = magnet_dir[iatom];
				}
				
				auto & ps = pseudo_for_element(ions.species(iatom));
				
				std::array<double, 4> polarization{};
				polarization[0] = 1.0;
				if (nspin > 1) polarization = observables::sdm_oriented(atomic_magnet);

				if(ps.has_electronic_density()){
					basis::spherical_grid sphere(basis, atom_position, ps.electronic_density_radius());
					gpu::run(sphere.size(),
									 [dens = begin(density.hypercubic()), sph = sphere.ref(), spline = ps.electronic_density().function(), pol = polarization, nspin] GPU_LAMBDA (auto ipoint){
										 auto rr = sph.distance(ipoint);
										 auto density_val = spline(rr);
										 auto pol = polarization;
										 if(ispin > 1) return;
										 if(ispin == 1) pol = 1.0 - pol;
										 gpu::atomic(dens[sph.grid_point(ipoint)[0]][sph.grid_point(ipoint)[1]][sph.grid_point(ipoint)[2]][ispin]) += pol*density_val;
									 });

				} else {

					//just some crude guess for now
					basis::spherical_grid sphere(basis, atom_position, 3.0);
					
					gpu::run(nspin, sphere.size(),
									 [dens = begin(density.hypercubic()), sph = sphere.ref(), zval = ps.valence_charge(), pol = polarization, nspin] GPU_LAMBDA (auto ispin, auto ipoint){
										 auto rr = sph.distance(ipoint);
										 auto pol = polarization;
										 if(ispin == 1) pol = 1.0 - pol;
										 gpu::atomic(dens[sph.grid_point(ipoint)[0]][sph.grid_point(ipoint)[1]][sph.grid_point(ipoint)[2]][ispin]) += pol*zval/(M_PI)*exp(-2.0*rr);
									 });
					
				}
			}

			density.all_reduce(comm);
			return density;
		}

		////////////////////////////////////////////////////////////////////////////////////

		auto has_overlap() const {
			return has_overlap_;
		}

		////////////////////////////////////////////////////////////////////////////////////

		auto has_nlcc() const {
			return has_nlcc_;
		}

		////////////////////////////////////////////////////////////////////////////////////

		template <class CommType, class basis_type, class ions_type>
		basis::field<basis_type, double> nlcc_density(CommType & comm, const basis_type & basis, const ions_type & ions) const {

			CALI_CXX_MARK_FUNCTION;

			parallel::partition part(ions.size(), comm);
			
			basis::field<basis_type, double> density(basis);

			density.fill(0.0);
			
			for(auto iatom = part.start(); iatom < part.end(); iatom++){
				
				auto atom_position = ions.positions()[iatom];
				
				auto & ps = pseudo_for_element(ions.species(iatom));

				if(not ps.has_nlcc_density()) continue;

				assert(has_nlcc());
				
				basis::spherical_grid sphere(basis, atom_position, ps.nlcc_density_radius());

				gpu::run(sphere.size(),
								 [dens = begin(density.cubic()),
									sph = sphere.ref(),
									spline = ps.nlcc_density().function()] GPU_LAMBDA (auto ipoint){
									 auto rr = sph.distance(ipoint);
									 auto density_val = spline(rr);
									 gpu::atomic(dens[sph.grid_point(ipoint)[0]][sph.grid_point(ipoint)[1]][sph.grid_point(ipoint)[2]]) += density_val;
								 });
				
			}

			density.all_reduce(comm);
			return density;			
		}

		////////////////////////////////////////////////////////////////////////////////////

		template <class Comm, class Ions>
		gpu::array<vector3<double>, 1> nlcc_forces(Comm & comm, basis::field_set<basis::real_space, double> const & vxc, Ions const & ions) const {
			CALI_CXX_MARK_FUNCTION;

			// NLCC Forces from Kronik et al., J. Chem. Phys. 115, 4322 (2001): Eq. 9

			gpu::array<vector3<double>, 1> forces(ions.size(), {0.0, 0.0, 0.0});

			auto && basis = vxc.basis();
			
			parallel::partition part(ions.size(), comm);
			
			for(auto iatom = part.start(); iatom < part.end(); iatom++){
				
				auto atom_position = ions.positions()[iatom];
				
				auto & ps = pseudo_for_element(ions.species(iatom));
				
				if(not ps.has_nlcc_density()) continue;

				assert(has_nlcc());
				
				basis::spherical_grid sphere(basis, atom_position, 1.1*ps.nlcc_density_radius());

				auto ff = gpu::run(gpu::reduce(sphere.size()), zero<vector3<double, contravariant>>(),
													 [vx = begin(vxc.hypercubic()), nspin = std::min(2l, vxc.local_set_size()), sph = sphere.ref(), spline = ps.nlcc_density().function()]
													 GPU_LAMBDA (auto ipoint){
														 auto vv = 0.0;
														 for(int ispin = 0; ispin < nspin; ispin++) vv += vx[sph.grid_point(ipoint)[0]][sph.grid_point(ipoint)[1]][sph.grid_point(ipoint)[2]][ispin];
														 auto rr = sph.distance(ipoint);
														 if(rr < 1e-15) return zero<vector3<double, contravariant>>();
														 auto grad = spline.derivative(rr)*sph.point_pos(ipoint)/rr;
														 return vv*grad;
													 });

				forces[iatom] = basis.volume_element()*basis.cell().to_cartesian(ff);
			}

			if(basis.comm().size() > 1) {
				basis.comm().all_reduce_in_place_n(reinterpret_cast<double *>(raw_pointer_cast(forces.data_elements())), 3*forces.size());
			}

			//we should use allgather here
			if(comm.size() > 1) {
				comm.all_reduce_in_place_n(reinterpret_cast<double *>(raw_pointer_cast(forces.data_elements())), 3*forces.size());
			}
			
			return forces;
		}

		////////////////////////////////////////////////////////////////////////////////////
		
		template <class output_stream>
		void info(output_stream & out) const {
			out << "ATOMIC POTENTIAL:" << std::endl;
			out << "	Number of species		= " << num_species() << std::endl;
			out << std::endl;
		}

		auto & range_separation() const {
			return sep_;
		}

		auto & double_grid() const {
			return double_grid_;
		}

	};

}
}
#endif

#ifdef INQ_HAMILTONIAN_ATOMIC_POTENTIAL_UNIT_TEST
#undef INQ_HAMILTONIAN_ATOMIC_POTENTIAL_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;
	using ionic::species;

	double const gcut = 0.785;

	parallel::communicator comm{boost::mpi3::environment::get_self_instance()};
		
	SECTION("Non-existing element"){
		ionic::species_set el_list;
		el_list.insert("P");
		el_list.insert("X");
	
		CHECK_THROWS(hamiltonian::atomic_potential(el_list, gcut));
	}

	SECTION("Empty list"){
		ionic::species_set el_list;

		hamiltonian::atomic_potential pot(el_list, gcut);

		CHECK(pot.num_species() == 0);

		CHECK(not pot.has_nlcc());
		
	}

	SECTION("CNOH"){
		ionic::species_set el_list;
		el_list.insert("C");
		el_list.insert("N");
		el_list.insert("O");
		el_list.insert("H");

		hamiltonian::atomic_potential pot(el_list, gcut);

		CHECK(pot.num_species() == 4);
	}

	SECTION("Construct from a geometry"){

		auto ions = systems::ions::parse(config::path::unit_tests_data() + "benzene.xyz", systems::cell::cubic(20.0_b));
		
		basis::real_space rs(ions.cell(), /*spacing = */ 0.49672941, comm);
		
		hamiltonian::atomic_potential pot(ions.species_list(), rs.gcutoff());
		
		CHECK(pot.num_species() == 2);
		CHECK(pot.num_electrons(ions.symbols()) == 30.0_a);
		
		auto vv = pot.local_potential(comm, rs, ions);
		
		CHECK(operations::integral(vv) == -45.5744357466_a);
		
		CHECK(vv.cubic()[5][3][0] == -1.6226427555_a);
		CHECK(vv.cubic()[3][1][0] == -0.2739253316_a);
		
		auto id = pot.ionic_density(comm, rs, ions);
		
		CHECK(operations::integral(id) == -30.0000000746_a);
		CHECK(id.cubic()[5][3][0] == -0.9448936487_a);
		CHECK(id.cubic()[3][1][0] == -0.2074502252_a);

		states::ks_states unp(states::spin_config::UNPOLARIZED, 11.0);
		
		auto nn_unp = pot.atomic_electronic_density(comm, rs, ions, unp);

		CHECK(nn_unp.set_size() == 1);		
		CHECK(operations::integral_sum(nn_unp) == 29.9562520003_a);
		CHECK(nn_unp.hypercubic()[5][3][0][0] == 0.1330589609_a);
		CHECK(nn_unp.hypercubic()[3][1][0][0] == 0.1846004508_a);

		states::ks_states pol(states::spin_config::POLARIZED, 11.0);
		
		auto nn_pol = pot.atomic_electronic_density(comm, rs, ions, pol);

		CHECK(nn_pol.set_size() == 2);
		CHECK(operations::integral_sum(nn_pol) == 29.9562519176_a);
		CHECK(nn_pol.hypercubic()[5][3][0][0] == Approx(1.2*0.066529473));
		CHECK(nn_pol.hypercubic()[3][1][0][0] == Approx(1.2*0.0923002217));
		CHECK(nn_pol.hypercubic()[5][3][0][1] == Approx(0.8*0.066529473));
		CHECK(nn_pol.hypercubic()[3][1][0][1] == Approx(0.8*0.0923002217));
		
		CHECK(pot.has_nlcc());
		
		auto nlcc = pot.nlcc_density(comm, rs, ions);
		
		CHECK(operations::integral(nlcc) == 3.0083012065_a);
		CHECK(nlcc.cubic()[5][3][0] == 0.6248217151_a);
		CHECK(nlcc.cubic()[3][1][0] == 0.0007040027_a);
	
	}
	
}
#endif
