/* -*- indent-tabs-mode: t -*- */

#ifndef SYSTEMS__ELECTRONS
#define SYSTEMS__ELECTRONS

// Copyright (C) 2019-2025 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include <basis/field_set.hpp>
#include <basis/real_space.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <states/ks_states.hpp>
#include <hamiltonian/ks_hamiltonian.hpp>
#include <hamiltonian/energy.hpp>
#include <interface/actions.hpp>
#include <ionic/brillouin.hpp>
#include <ionic/interaction.hpp>
#include <observables/density.hpp>
#include <operations/randomize.hpp>
#include <operations/integral.hpp>
#include <operations/io.hpp>
#include <operations/orthogonalize.hpp>
#include <math/complex.hpp>
#include <input/kpoints.hpp>
#include <options/electrons.hpp>
#include <options/theory.hpp>
#include <systems/ions.hpp>
#include <states/orbital_set.hpp>
#include <utils/num_str.hpp>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp> // uuids::random_generator

#include <cfloat>

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/transform_width.hpp>

#include<spdlog/spdlog.h>
#include<spdlog/sinks/stdout_color_sinks.h>
#include<spdlog/fmt/ostr.h> // print user defined types

#include <utils/profiling.hpp>

namespace inq {
namespace systems {

class electrons {

public:

	using kpin_type = std::vector<states::orbital_set<basis::real_space, complex>>;

private:
	
	inq::ionic::brillouin brillouin_zone_;
	hamiltonian::atomic_potential atomic_pot_;
	states::ks_states states_;
	mutable parallel::cartesian_communicator<3> full_comm_;
	mutable parallel::cartesian_communicator<1> kpin_comm_;
	mutable parallel::cartesian_communicator<2> kpin_states_comm_;
	mutable parallel::cartesian_communicator<1> states_comm_;
	mutable parallel::cartesian_communicator<2> states_basis_comm_;
	basis::real_space states_basis_;
	basis::real_space density_basis_;
	kpin_type kpin_;
	gpu::array<double, 2> eigenvalues_;
	gpu::array<double, 2> occupations_;
	gpu::array<double, 1> kpin_weights_;
	long max_local_set_size_;
	long max_local_spinor_set_size_;
	basis::field_set<basis::real_space, double> spin_density_;
	std::shared_ptr<spdlog::logger> logger_;
	parallel::partition kpin_part_;
	
public:
	
	static auto kpin_subcomm(parallel::cartesian_communicator<3> & comm){
		return comm.axis(input::parallelization::dimension_kpoints());
	}
	static auto states_subcomm(parallel::cartesian_communicator<3> & comm){
		return comm.axis(input::parallelization::dimension_states());
	}
	static auto basis_subcomm(parallel::cartesian_communicator<3> & comm){
		return comm.axis(input::parallelization::dimension_domains());
	}
	static auto states_basis_subcomm(parallel::cartesian_communicator<3> & comm){
		assert(input::parallelization::dimension_domains() < input::parallelization::dimension_states());		
		return comm.plane(input::parallelization::dimension_domains(), input::parallelization::dimension_states());
	}
	static auto kpin_states_subcomm(parallel::cartesian_communicator<3> & comm){
		assert(input::parallelization::dimension_states() < input::parallelization::dimension_kpoints());
		return comm.plane(input::parallelization::dimension_states(), input::parallelization::dimension_kpoints());
	}
	
	auto & kpin() const {
		return kpin_;
	}

	auto & kpin() {
		return kpin_;
	}

	auto & occupations() const {
		return occupations_;
	}

	auto & occupations() {
		return occupations_;
	}

	template <typename KptsType = input::kpoints::list>
	electrons(const inq::systems::ions & ions, KptsType const & kpts, const options::electrons & conf = {}):
		electrons(input::environment::global().par(), ions, conf, kpts)
	{
	}

	template <typename KptsType = input::kpoints::list>	
	electrons(const inq::systems::ions & ions, const options::electrons & conf = {}, KptsType const & kpts = input::kpoints::gamma()):
		electrons(input::environment::global().par(), ions, conf, kpts)
	{
	}
	
	template <typename KptsType = input::kpoints::list>
	electrons(input::parallelization const & dist, const inq::systems::ions & ions, KptsType const & kpts, const options::electrons & conf = {}):
		electrons(dist, ions, conf, kpts)
	{
	}

	template <typename KptsType = input::kpoints::list>	
	electrons(input::parallelization const & dist, const inq::systems::ions & ions, const options::electrons & conf = {}, KptsType const & kpts = input::kpoints::gamma()):
		brillouin_zone_(ions, kpts),
		atomic_pot_(ions.species_list(), basis::real_space::gcutoff(ions.cell(), conf.spacing_value()), conf),
		states_(conf.spin_val(), atomic_pot_.num_electrons(ions.symbols()) + conf.extra_electrons_val(), conf.extra_states_val(), conf.temperature_val(), kpts.size()),
		full_comm_(dist.cart_comm(conf.num_spin_components_val(), brillouin_zone_.size(), states_.num_states())),
		kpin_comm_(kpin_subcomm(full_comm_)),
		kpin_states_comm_(kpin_states_subcomm(full_comm_)),
		states_comm_(states_subcomm(full_comm_)),
		states_basis_comm_(states_basis_subcomm(full_comm_)),
		states_basis_(ions.cell(), conf.spacing_value(), basis_subcomm(full_comm_)),
		density_basis_(states_basis_.refine(conf.density_factor_value())),
		spin_density_(density_basis_, states_.num_density_components()),
		kpin_part_(kpts.size()*states_.num_spin_indices(), kpin_comm_)
	{
		CALI_CXX_MARK_FUNCTION;

		assert(kpin_part_.local_size() > 0);
		assert(density_basis_.comm().size() == states_basis_.comm().size());

		auto nproc_spin = 1;
		if(states_.num_spin_indices() == 2 and kpin_comm_.size()%2 == 0) nproc_spin = 2;

		parallel::cartesian_communicator<2> spin_kpoints_comm(kpin_comm_, {nproc_spin, boost::mpi3::fill});

		parallel::partition spin_part(states_.num_spin_indices(), spin_kpoints_comm.axis(0));
		parallel::partition kpts_part(kpts.size(), spin_kpoints_comm.axis(1));

		assert(kpin_part_.local_size() == kpts_part.local_size()*spin_part.local_size()); //this is always true because the spin size is either 1 or 2
		
		kpin_weights_.reextent({kpin_part_.local_size()});

		max_local_set_size_ = 0;
		max_local_spinor_set_size_ = 0;
		auto ilot = 0;
		for(int ispin = 0; ispin < spin_part.local_size(); ispin++){
			for(int ikpt = 0; ikpt < kpts_part.local_size(); ikpt++){
				kpin_weights_[ilot] = brillouin_zone_.kpoint_weight(kpts_part.local_to_global(ikpt).value());
				auto kpoint = brillouin_zone_.kpoint(kpts_part.local_to_global(ikpt).value());
				kpin_.emplace_back(states_basis_, states_.num_states(), states_.spinor_dim(), kpoint, spin_part.local_to_global(ispin).value(), states_basis_comm_);
				max_local_set_size_ = std::max(max_local_set_size_, kpin_[ikpt].local_set_size());
				max_local_spinor_set_size_ = std::max(max_local_spinor_set_size_, kpin_[ikpt].local_spinor_set_size());
				ilot++;
			}
		}

		assert(long(kpin_.size()) == kpin_part_.local_size());
		assert(max_local_set_size_ > 0);
		
		eigenvalues_.reextent({static_cast<boost::multi::size_t>(kpin_.size()), max_local_spinor_set_size_});
		occupations_.reextent({static_cast<boost::multi::size_t>(kpin_.size()), max_local_spinor_set_size_});

		if(states_.num_electrons() == 0.0) throw std::runtime_error("inq error: the system does not have any electrons");
		if(states_.num_electrons() <  0.0) throw std::runtime_error("inq error: the system has a negative number of electrons");
		
		print(ions);

	}

	electrons(electrons && old_el, input::parallelization const & new_dist):
		brillouin_zone_(std::move(old_el.brillouin_zone_)),
		atomic_pot_(std::move(old_el.atomic_pot_)),
		states_(std::move(old_el.states_)),
		full_comm_(new_dist.cart_comm(old_el.states_.num_spin_indices(), brillouin_zone_.size(), states_.num_states())),
		kpin_comm_(kpin_subcomm(full_comm_)),
		kpin_states_comm_(kpin_states_subcomm(full_comm_)),
		states_comm_(states_subcomm(full_comm_)),
		states_basis_comm_(states_basis_subcomm(full_comm_)),
		states_basis_(std::move(old_el.states_basis_), basis_subcomm(full_comm_)),
		density_basis_(std::move(old_el.density_basis_), basis_subcomm(full_comm_)),
		kpin_weights_(std::move(old_el.kpin_weights_)),
		max_local_set_size_(std::move(old_el.max_local_set_size_)),
		max_local_spinor_set_size_(std::move(old_el.max_local_spinor_set_size_)),
		spin_density_(std::move(old_el.spin_density_), {density_basis_.comm(), {density_basis_.comm().size(), 1}}),
		logger_(std::move(old_el.logger_)),
		kpin_part_(std::move(old_el.kpin_part_))
	{

		assert(kpin_comm_ == old_el.kpin_comm_); //resizing of k points not supported for the moment

		max_local_set_size_ = 0;
		max_local_spinor_set_size_ = 0;
		for(auto & oldphi : old_el.kpin_){
			kpin_.emplace_back(std::move(oldphi), states_basis_comm_);
			max_local_set_size_ = std::max(max_local_set_size_, kpin_.back().local_set_size());
			max_local_spinor_set_size_ = std::max(max_local_spinor_set_size_, kpin_.back().local_set_size());
		}

		assert(kpin_.size() == old_el.kpin_.size());

		eigenvalues_.reextent({static_cast<boost::multi::size_t>(kpin_.size()), max_local_spinor_set_size_});
		occupations_.reextent({static_cast<boost::multi::size_t>(kpin_.size()), max_local_spinor_set_size_});
		
		for(unsigned ilot = 0; ilot < kpin_.size(); ilot++){

			parallel::partition part(kpin_[ilot].spinor_set_size(), states_subcomm(old_el.full_comm_));
			
			parallel::array_iterator eigit(part, states_subcomm(old_el.full_comm_), +old_el.eigenvalues_[ilot]);
			parallel::array_iterator occit(part, states_subcomm(old_el.full_comm_), +old_el.occupations_[ilot]);

			for(; eigit != eigit.end(); ++eigit){
 				
				for(int ist = 0; ist < eigenvalues_[ilot].size(); ist++){
					auto istg = kpin_[ilot].spinor_set_part().local_to_global(ist);
					if(part.contains(istg.value())){
						eigenvalues_[ilot][ist] = (*eigit)[part.global_to_local(istg)];
						occupations_[ilot][ist] = (*occit)[part.global_to_local(istg)];
					}
				}
				
				++occit;
			}
		}
	}

	void print(const inq::systems::ions & ions){
		if(full_comm_.root()){
			logger_ = spdlog::stdout_color_mt("electrons:"+ generate_tiny_uuid());
			logger_->set_level(spdlog::level::trace);
		}

		if(logger()){
			logger()->info("System information:\n\n{}{}{}{}{}{}", ions.cell(), ions.species_list(), ions, brillouin_zone_, states_basis_, states_);
			if(ions.size() > 0) logger()->info("system symmetries: " + ions.symmetry_string());
		}

		auto myid = gpu::id();
		auto gpuids = full_comm_.all_gather_as<boost::multi::array<decltype(myid), 1>>(myid);

		basis::fourier_space fourier_basis(states_basis_);
			
		if(logger()){
			logger()->info("parallelization:");
			logger()->info("  electrons divided among {} processes ({} kpoints x {} states x {} domains)", full_comm_.size(),
										 full_comm_.shape()[input::parallelization::dimension_kpoints()],
										 full_comm_.shape()[input::parallelization::dimension_states()],
										 full_comm_.shape()[input::parallelization::dimension_domains()]);
#ifdef ENABLE_GPU
			for(int iproc = 0; iproc < full_comm_.size(); iproc++){
				logger()->info("  process {} has gpu id {}", iproc, gpuids[iproc]);
			}
#else
			logger()->info("  inq is running on the cpu\n");
#endif
			logger()->info("k-point parallelization:");
			logger()->info("  {} k-points/spin indices divided among {} partitions", kpin_part_.size(), kpin_part_.comm_size());
			logger()->info("  partition 0 has {} k-points/spin indices and the last partition has {}\n", kpin_part_.local_size(0), kpin_part_.local_size(kpin_part_.comm_size() - 1));

			if(kpin_part_.local_size(kpin_part_.comm_size() - 1) == 0) {
				interface::actions::error(full_comm_, "A processor has 0 kpoints/spin indices assigned, please change the number of processors or the parallelization distribution.");
			}
			
			logger()->info("state parallelization:");
			logger()->info("  {} states divided among {} partitions", kpin()[0].set_part().size(), kpin()[0].set_part().comm_size());
			logger()->info("  partition 0 has {} states and the last partition has {} states\n", kpin()[0].set_part().local_size(0), kpin()[0].set_part().local_size(kpin()[0].set_part().comm_size() - 1));

			if(kpin()[0].set_part().local_size(kpin()[0].set_part().comm_size() - 1) == 0) {
				interface::actions::error(full_comm_, "A processor has 0 states assigned, please change the number of processors or the parallelization distribution.");
			}
			
			logger()->info("real-space parallelization:");
			logger()->info("  {} slices ({} points) divided among {} partitions", states_basis_.cubic_part(1).size(), states_basis_.part().size(), states_basis_.cubic_part(1).comm_size());
			logger()->info("  partition 0 has {} slices and the last partition has {} slices ({} and {} points)",
										 states_basis_.cubic_part(1).local_size(0), states_basis_.cubic_part(1).local_size(states_basis_.part().comm_size() - 1),
										 states_basis_.part().local_size(0), states_basis_.part().local_size(states_basis_.part().comm_size() - 1));

			if(states_basis_.part().local_size(states_basis_.part().comm_size() - 1) == 0) {
				interface::actions::error(full_comm_, "A processor has 0 real-space points assigned, please change the number of processors or the parallelization distribution.");
			}
			
			logger()->info("fourier-space parallelization:");
			logger()->info("  {} slices ({} points) divided among {} partitions", fourier_basis.cubic_part(0).size(), fourier_basis.part().size(), fourier_basis.cubic_part(0).comm_size());
			logger()->info("  partition 0 has {} slices and the last partition has {} slices ({} and {} points)\n",
										 fourier_basis.cubic_part(0).local_size(0), fourier_basis.cubic_part(0).local_size(fourier_basis.part().comm_size() - 1),
										 fourier_basis.part().local_size(0), fourier_basis.part().local_size(fourier_basis.part().comm_size() - 1));

			if(fourier_basis.part().local_size(fourier_basis.part().comm_size() - 1) == 0) {
				interface::actions::error(full_comm_, "A processor has 0 real-space points assigned, please change the number of processors or the parallelization distribution.");
			}

		}
	}

	template <typename ArrayType>
	void update_occupations(ArrayType const& eigenval) {
		states_.update_occupations(kpin_states_comm_, eigenval, kpin_weights_, occupations_);
	}

	void save(std::string const & dirname) const {

		full_comm_.barrier();
		if(logger()) logger()->info("saving electrons to '{}/'", dirname);
		full_comm_.barrier();

		int iphi = 0;
		for(auto & phi : kpin()){
			auto basedir = dirname + "/kpin" + utils::num_to_str(iphi + kpin_part_.start());
			operations::io::save(basedir + "/states", phi);
			assert(occupations()[iphi].size() == kpin()[iphi].spinor_set_part().local_size());
			if(states_basis_.comm().root()) operations::io::save(basedir + "/occupations", states_comm_, kpin()[iphi].spinor_set_part(), +occupations()[iphi]);
			iphi++;
		}

		if(kpin_states_comm_.root()) operations::io::save(dirname + "/spin_density", spin_density_);

		full_comm_.barrier();
		if(logger()) logger()->info("  saving done.", dirname);
		full_comm_.barrier();

	}
		
	auto try_load(std::string const & dirname) {
		auto success = true;

		full_comm_.barrier();
		if(logger()) logger()->info("loading electrons from '{}/'", dirname);
		full_comm_.barrier();

		int iphi = 0;
		for(auto & phi : kpin()){
			auto basedir = dirname + "/kpin" + utils::num_to_str(iphi + kpin_part_.start());
			success = success and operations::io::load(basedir + "/states", phi);

			gpu::array<double, 1> tmpocc(kpin()[iphi].spinor_set_part().local_size());
			success = success and operations::io::load(basedir + "/occupations", states_comm_, kpin()[iphi].spinor_set_part(), tmpocc);
			occupations()[iphi] = tmpocc;
			
			iphi++;
		}

		success = success and operations::io::load(dirname + "/spin_density",	spin_density_);

		full_comm_.all_reduce_in_place_n(&success, 1, std::logical_and<>{});
		
		if(success) {
			if(logger()) logger()->info("  loading successful");
		} else {
			if(logger()) logger()->info("  loading failed");
		}

		return success;
	}

	void load(std::string const & dirname) {
		auto success = try_load(dirname);
		if(not success) throw std::runtime_error("\n\nERROR: INQ CAN'T LOAD RESTART INFORMATION FROM THE PATH '" + dirname + "'.\n");
	}
	
	auto & eigenvalues() const {
		return eigenvalues_;
	}

	auto & eigenvalues() {
		return eigenvalues_;
	}

	long kpin_size() const {
		return kpin().size();
	}

	auto & kpin_weights() const {
		return kpin_weights_;
	}

	auto & kpin_part() const {
		return kpin_part_;
	}

	auto & max_local_set_size() const {
		return max_local_set_size_;
	}

	auto & max_local_spinor_set_size() const {
		return max_local_spinor_set_size_;
	}

	auto density() const {
		return observables::density::total(spin_density_);
	}

	auto & spin_density() const {
		return spin_density_;
	}

	auto & spin_density() {
		return spin_density_;
	}

	auto & states() const {
		return states_;
	}

	auto root() const {
		return full_comm_.root();
	}

	auto kpoint_index(states::orbital_set<basis::real_space, complex> const & phi) const {

		CALI_CXX_MARK_FUNCTION;
		
		for(auto ik = 0; ik < brillouin_zone_.size(); ik++){
			if(phi.basis().cell().norm(phi.kpoint() - brillouin_zone_.kpoint(ik)) < 1e-10) return ik;
		}
		assert(false);
		return 0;
	}

	auto & brillouin_zone() const {
		return brillouin_zone_;
	}

	auto & atomic_pot() const {
		return atomic_pot_;
	}

	auto & states_basis() const {
		return states_basis_;
	}

	auto & density_basis() const {
		return density_basis_;
	}

	auto & full_comm() const {
		return full_comm_;
	}
	
	auto & kpin_comm() const {
		return kpin_comm_;
	}
	
	auto & kpin_states_comm() const {
		return kpin_states_comm_;
	}
	
	auto & states_comm() const {
		return states_comm_;
	}
	
	auto & states_basis_comm() const {
		return states_basis_comm_;
	}

	std::shared_ptr<spdlog::logger> const& logger() const{
		return logger_;
	}

private:
	
	static std::string generate_tiny_uuid(){
		auto uuid = boost::uuids::random_generator{}();
		uint32_t tiny = hash_value(uuid) % std::numeric_limits<uint32_t>::max();
		using namespace boost::archive::iterators;
		using it = base64_from_binary<transform_width<unsigned char*, 6, 8>>;
		return std::string(it((unsigned char*)&tiny), it((unsigned char*)&tiny+sizeof(tiny)));//.append((3-sizeof(tiny)%3)%3,'=');
	}
	
};

}
}
#endif

#ifdef INQ_SYSTEMS_ELECTRONS_UNIT_TEST
#undef INQ_SYSTEMS_ELECTRONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	
	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	systems::ions ions(systems::cell::orthorhombic(6.0_b, 6.0_b, 1.0_b));

	ions.insert("Cu", {0.0_b,  0.0_b,  0.0_b});
	ions.insert("Cu", {1.0_b,  0.0_b,  0.0_b});
	
	auto par = input::parallelization(comm);
	
	systems::electrons electrons(par, ions, options::electrons{}.cutoff(15.0_Ha));

	CHECK(electrons.states().num_electrons() == 38.0_a);
	CHECK(electrons.states().num_states() == 19);

	int iphi = 0;
	for(auto & phi : electrons.kpin()) {
		
		for(int ist = 0; ist < phi.set_part().local_size(); ist++){
			auto istg = phi.set_part().local_to_global(ist);
			
			electrons.occupations()[iphi][ist] = cos(istg.value());
			
			for(int ip = 0; ip < phi.basis().local_size(); ip++){
				auto ipg = phi.basis().part().local_to_global(ip);
				phi.matrix()[ip][ist] = 20.0*(ipg.value() + 1)*sqrt(istg.value());
			}
		}
		iphi++;
	}
	
	electrons.spin_density() = observables::density::calculate(electrons);
	CHECK(operations::integral_sum(electrons.spin_density()) == -1315151005.0813348293_a);
	
	electrons.save("electron_restart");
	
	systems::electrons electrons_read(par, ions, options::electrons{}.cutoff(15.0_Ha));
	
	electrons_read.load("electron_restart");

	CHECK(operations::integral_sum(electrons_read.spin_density()) == -1315151005.0813348293_a);
	CHECK(operations::integral_sum(electrons.spin_density()) == Catch::Approx(operations::integral_sum(electrons.spin_density())));
	CHECK(electrons.kpin_size() == electrons_read.kpin_size());
	
	iphi = 0;
	for(auto & phi : electrons.kpin()) {
		
		for(int ist = 0; ist < phi.set_part().local_size(); ist++){
			CHECK(electrons.occupations()[iphi][ist] == electrons_read.occupations()[0][ist]);
			for(int ip = 0; ip < phi.basis().local_size(); ip++){
				CHECK(phi.matrix()[ip][ist] == electrons_read.kpin()[iphi].matrix()[ip][ist]);
			}
		}
		iphi++;
	}
	
	SECTION("Redistribute"){
		
		systems::electrons newel(std::move(electrons_read), input::parallelization(comm).domains(1).states());

		CHECK(newel.kpin()[0].spinor_dim() == 1);
			
		newel.save("newel_restart");
		
		systems::electrons newel_read(par, ions, options::electrons{}.cutoff(15.0_Ha));
		newel_read.load("newel_restart");
		
		CHECK(electrons.kpin_size() == newel_read.kpin_size());
		
		iphi = 0;
		for(auto & phi : electrons.kpin()) {
			
			for(int ist = 0; ist < phi.set_part().local_size(); ist++){
				CHECK(electrons.occupations()[iphi][ist] == newel_read.occupations()[0][ist]);
				for(int ip = 0; ip < phi.basis().local_size(); ip++){
					CHECK(phi.matrix()[ip][ist] == newel_read.kpin()[iphi].matrix()[ip][ist]);
				}
			}
			iphi++;
		}

	}

	SECTION("Restart errors"){
		CHECK(not electrons.try_load("directory_that_doesnt_exist"));
		CHECK_THROWS(electrons.load("directory_that_doesnt_exist"));
	}

	SECTION("Electrons with kpoints"){
		systems::electrons electrons(par, ions, input::kpoints::grid({2, 1, 1}, true), options::electrons{}.cutoff(15.0_Ha));
		
		CHECK(electrons.states().num_electrons() == 38.0_a);
		CHECK(electrons.states().num_states() == 19);
		
		for(auto & phi : electrons.kpin()) {
			
			auto kpoint_index = electrons.kpoint_index(phi);

			CHECK(phi.kpoint()[0] == electrons.brillouin_zone().kpoint(kpoint_index)[0]);
			CHECK(phi.kpoint()[1] == electrons.brillouin_zone().kpoint(kpoint_index)[1]);
			CHECK(phi.kpoint()[2] == electrons.brillouin_zone().kpoint(kpoint_index)[2]);
		}
		
	}
}
#endif

