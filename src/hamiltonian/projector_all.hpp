/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__PROJECTOR_ALL
#define INQ__HAMILTONIAN__PROJECTOR_ALL

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <gpu/array.hpp>
#include <math/vector3.hpp>
#include <basis/real_space.hpp>
#include <basis/spherical_grid.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <matrix/lu.hpp>
#include <utils/profiling.hpp>
#include <utils/raw_pointer_cast.hpp>

#ifdef ENABLE_HIP
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define cublasDgemmStridedBatched hipblasDgemmStridedBatched
#endif

namespace inq {
namespace hamiltonian {

class projector_all {

	int nprojs_;
	long max_sphere_size_;
	int max_nlm_;
	gpu::array<vector3<int>, 2> points_;
	gpu::array<vector3<float, contravariant>, 2> positions_;
	gpu::array<double, 3> coeff_;
	gpu::array<double, 3> overlap_coeff_;
	gpu::array<double, 3> matrices_;
	gpu::array<int, 1> nlm_;
	gpu::array<int, 1> iatom_;
	gpu::array<bool, 1> locally_empty_;

public: // for CUDA
	
	template <typename ProjectorsType>
	void constructor(ProjectorsType const & projectors){

		CALI_CXX_MARK_FUNCTION;

		max_sphere_size_ = 0;
		max_nlm_ = 0;
		for(auto it = projectors.cbegin(); it != projectors.cend(); ++it) {
			max_sphere_size_ = std::max(max_sphere_size_, it->sphere_.size());
			max_nlm_ = std::max(max_nlm_, it->nproj_);			
		}

		points_ = decltype(points_)({nprojs_, max_sphere_size_});
		positions_ = decltype(positions_)({nprojs_, max_sphere_size_});		
    coeff_ = decltype(coeff_)({nprojs_, max_nlm_, max_nlm_}, 0.0);
    overlap_coeff_ = decltype(overlap_coeff_)({nprojs_, max_nlm_, max_nlm_}, 0.0);
    matrices_ = decltype(matrices_)({nprojs_, max_nlm_, max_sphere_size_});
		
    auto iproj = 0;
    for(auto it = projectors.cbegin(); it != projectors.cend(); ++it) {
			gpu::run(max_sphere_size_,
							 [poi = begin(points_), pos = begin(positions_), sph = it->sphere_.ref(), iproj, npoint = it->sphere_.size()] GPU_LAMBDA (auto ipoint){
								 if(ipoint < unsigned (npoint)){
									 poi[iproj][ipoint] = sph.grid_point(ipoint);	
									 pos[iproj][ipoint] = sph.point_pos(ipoint);							 
								 } else {
									 poi[iproj][ipoint] = {-1, -1, -1};
								 }
							 });
			
			gpu::run(max_sphere_size_, max_nlm_,
							 [mat = begin(matrices_), itmat = begin(it->matrix_), iproj, np = it->sphere_.size(), nlm = it->nproj_] GPU_LAMBDA (auto ipoint, auto ilm){
								 if(ipoint < (unsigned) np and ilm < (unsigned) nlm) {
									 mat[iproj][ilm][ipoint] = itmat[ilm][ipoint];
								 } else {
									 mat[iproj][ilm][ipoint] = 0.0;								 
								 }
							 });

			coeff_[iproj]({0, it->nproj_}, {0, it->nproj_}) = it->kb_coeff_;
			overlap_coeff_[iproj]({0, it->nproj_}, {0, it->nproj_}) = it->overlap_mat_;

			nlm_[iproj] = it->nproj_;
			iatom_[iproj] = it->iatom_;
			locally_empty_[iproj] = it->locally_empty();
			
      iproj++;
    }
    
	}
	
public:

	projector_all():
		nprojs_(0),
    max_sphere_size_(0),
    max_nlm_(0) {
  }
  
	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename ProjectorsType>
	projector_all(ProjectorsType const & projectors):
		nprojs_(projectors.size()),
		nlm_(nprojs_),
		iatom_(nprojs_),
		locally_empty_(nprojs_)
	{
		constructor(projectors);
	}

	////////////////////////////////////////////////////////////////////////////////////////////		
	
	template <typename Type, typename KpointType>
	gpu::array<Type, 3> gather(states::orbital_set<basis::real_space, Type> const & phi, KpointType const & kpoint) const {

		gpu::array<Type, 3> sphere_phi_all({nprojs_, max_sphere_size_, phi.local_set_size()});

		{ CALI_CXX_MARK_SCOPE("projector::gather");
				
			gpu::run(phi.local_set_size(), max_sphere_size_, nprojs_,
							 [sgr = begin(sphere_phi_all), gr = begin(phi.hypercubic()), poi = begin(points_), pos = begin(positions_), kpoint] GPU_LAMBDA (auto ist, auto ipoint, auto iproj){
								 if(poi[iproj][ipoint][0] >= 0){
									 auto phase = polar(1.0, dot(kpoint, pos[iproj][ipoint]));
									 sgr[iproj][ipoint][ist] = phase*gr[poi[iproj][ipoint][0]][poi[iproj][ipoint][1]][poi[iproj][ipoint][2]][ist];
								 } else {
									 sgr[iproj][ipoint][ist] = zero<Type>();
								 }
							 });
		}
		
		return sphere_phi_all;
	}

	////////////////////////////////////////////////////////////////////////////////////////////		
	
	template <typename KpointType>
	gpu::array<complex, 3> calculate_projections(states::orbital_set<basis::real_space, complex> const & phi, KpointType const & kpoint) const {
		
		auto sphere_phi_all = gather(phi, kpoint);
		gpu::array<complex, 3> projections_all({nprojs_, max_nlm_, phi.local_set_size()}, 0.0);
		
#ifndef ENABLE_GPU
		for(auto iproj = 0; iproj < nprojs_; iproj++){
			CALI_CXX_MARK_SCOPE("projector_gemm_1");

			if(locally_empty_[iproj]) continue;
			
			namespace blas = boost::multi::blas;
			blas::real_doubled(projections_all[iproj]) = blas::gemm(phi.basis().volume_element(), matrices_[iproj], blas::real_doubled(sphere_phi_all[iproj]));
		}
#else
		if(max_sphere_size_ > 0) {
			CALI_CXX_MARK_SCOPE("projector_gemm_1");			

			const double zero = 0.0;
			const double vol = phi.basis().volume_element();

			auto status = cublasDgemmStridedBatched(/*cublasHandle_t handle = */ boost::multi::cuda::cublas::context::get_instance().get(),
																							/*cublasOperation_t transa = */ CUBLAS_OP_N,
																							/*cublasOperation_t transb = */ CUBLAS_OP_N,
																							/*int m = */ 2*phi.local_set_size(),
																							/*int n = */ max_nlm_,
																							/*int k = */ max_sphere_size_,
																							/*const double *alpha = */ &vol,
																							/*const double *A = */ reinterpret_cast<double const *>(raw_pointer_cast(sphere_phi_all.data_elements())),
																							/*int lda = */ 2*phi.local_set_size(),
																							/*long long int strideA = */ 2*max_sphere_size_*phi.local_set_size(),
																							/*const double *B = */ raw_pointer_cast(matrices_.data_elements()),
																							/*int ldb = */ max_sphere_size_,
																							/*long long int strideB =*/ max_nlm_*max_sphere_size_,
																							/*const double *beta = */ &zero,
																							/*double *C = */ reinterpret_cast<double *>(raw_pointer_cast(projections_all.data_elements())),
																							/*int ldc = */ 2*phi.local_set_size(),
																							/*long long int strideC = */ 2*max_nlm_*phi.local_set_size(),
																							/*int batchCount = */ nprojs_);
			gpu::sync();
			
			assert(status == CUBLAS_STATUS_SUCCESS);
			
		}
#endif

		if(phi.basis().comm().size() > 1) {
			CALI_CXX_MARK_SCOPE("projector_all::project::reduce");
			phi.basis().comm().all_reduce_in_place_n(raw_pointer_cast(projections_all.data_elements()), projections_all.num_elements(), std::plus<>{});
		}

		return 	projections_all;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename Type, typename Coeff>
	void multiply_by_coefficients(gpu::array<Type, 3> & projections_all, Coeff const & coeff) const {
		
		CALI_CXX_MARK_FUNCTION;

		auto nst = get<2>(sizes(projections_all));

		gpu::array<Type, 3> dest(extensions(projections_all));
		
		gpu::run(nst, max_nlm_, nprojs_,
						 [proj = begin(projections_all), coe = begin(coeff), des = begin(dest), nlm = max_nlm_]
						 GPU_LAMBDA (auto ist, auto ilm, auto iproj){
							 auto acc = zero<Type>();
							 for(int jlm = 0; jlm < nlm; jlm++) acc += coe[iproj][ilm][jlm]*proj[iproj][jlm][ist];
							 des[iproj][ilm][ist] = acc;
						 });
		
		projections_all = std::move(dest);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename KpointType>
	gpu::array<complex, 3> project(states::orbital_set<basis::real_space, complex> const & phi, KpointType const & kpoint, bool apply_overlap = false) const {
		
		auto projections_all = calculate_projections(phi, kpoint);

		if (apply_overlap) {
			multiply_by_coefficients(projections_all, overlap_coeff_);
		} else {
			multiply_by_coefficients(projections_all, coeff_);
		}
		
		gpu::array<complex, 3> sphere_phi_all({nprojs_, max_sphere_size_, phi.local_set_size()});
		
#ifndef ENABLE_GPU
		for(auto iproj = 0; iproj < nprojs_; iproj++) {
			CALI_CXX_MARK_SCOPE("projector_gemm_2");

			if(locally_empty_[iproj]) continue;
			
			namespace blas = boost::multi::blas;
			blas::real_doubled(sphere_phi_all[iproj]) = blas::gemm(1., blas::T(matrices_[iproj]), blas::real_doubled(projections_all[iproj]));
		}
#else
		if(max_sphere_size_ > 0) {
			CALI_CXX_MARK_SCOPE("projector_gemm_2");

			const double zero = 0.0;
			const double one = 1.0;
			
			auto status = cublasDgemmStridedBatched(/*cublasHandle_t handle = */ boost::multi::cuda::cublas::context::get_instance().get(),
																							/*cublasOperation_t transa = */ CUBLAS_OP_N,
																							/*cublasOperation_t transb = */ CUBLAS_OP_T,
																							/*int m = */ 2*phi.local_set_size(),
																							/*int n = */ max_sphere_size_,
																							/*int k = */ max_nlm_,
																							/*const double *alpha = */ &one,
																							/*const double *A = */ reinterpret_cast<double const *>(raw_pointer_cast(projections_all.data_elements())),
																							/*int lda = */ 2*phi.local_set_size(),
																							/*long long int strideA = */ 2*max_nlm_*phi.local_set_size(),
																							/*const double *B = */ raw_pointer_cast(matrices_.data_elements()),
																							/*int ldb = */ max_sphere_size_,
																							/*long long int strideB =*/ max_nlm_*max_sphere_size_,
																							/*const double *beta = */ &zero,
																							/*double *C = */ reinterpret_cast<double *>(raw_pointer_cast(sphere_phi_all.data_elements())),
																							/*int ldc = */ 2*phi.local_set_size(),
																							/*long long int strideC = */ 2*max_sphere_size_*phi.local_set_size(),
																							/*int batchCount = */ nprojs_);

			gpu::sync();
			
			assert(status == CUBLAS_STATUS_SUCCESS);
			
		}
#endif

		return sphere_phi_all;
			
	}

	////////////////////////////////////////////////////////////////////////////////////////////		

	template <typename SpherePhiType, typename KpointType>
	void apply(SpherePhiType & sphere_vnlphi, states::orbital_set<basis::real_space, complex> & vnlphi, KpointType const & kpoint) const {

		CALI_CXX_MARK_SCOPE("projector_all::apply");

		gpu::run(vnlphi.local_set_size(), max_sphere_size_, nprojs_,
						 [sgr = begin(sphere_vnlphi), gr = begin(vnlphi.hypercubic()), poi = begin(points_), pos = begin(positions_), kpoint, empty = begin(locally_empty_)] GPU_LAMBDA (auto ist, auto ipoint, auto iproj){
							 if(not empty[iproj] and poi[iproj][ipoint][0] >= 0){
								 auto phase = polar(1.0, -dot(kpoint, pos[iproj][ipoint]));
								 gpu::atomic::add(&gr[poi[iproj][ipoint][0]][poi[iproj][ipoint][1]][poi[iproj][ipoint][2]][ist], phase*sgr[iproj][ipoint][ist]);
							 }
						 });
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename Proj, typename Coe>
  GPU_FUNCTION static auto energy_term(int const nlm, int const ist, int const ilm, int const iproj, Proj const & proj, Coe const & coe) {
		auto pp = proj[iproj][ilm][ist]; // the L diagonal values are 1.0
		auto qq = coe[iproj][ilm][ilm]*proj[iproj][ilm][ist];
		for(int jlm = ilm + 1; jlm < nlm; jlm++) {
			pp += coe[iproj][ilm][jlm]*proj[iproj][jlm][ist];
			qq += coe[iproj][jlm][ilm]*proj[iproj][jlm][ist];
		}
		
		return real(conj(pp)*qq);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename KpointType, typename Occupations>
	double energy(states::orbital_set<basis::real_space, complex> const & phi, KpointType const & kpoint, Occupations const & occupations, bool const reduce_states = true) const {

		auto projections_all = calculate_projections(phi, kpoint);
		
		auto en = gpu::run(gpu::reduce(phi.local_set_size()), gpu::reduce(max_nlm_), gpu::reduce(nprojs_), 0.0,
											 [proj = begin(projections_all), coe = begin(coeff_), occ = begin(occupations), spinor_size = phi.local_spinor_set_size(), nlm = max_nlm_]
											 GPU_LAMBDA (auto ist, auto ilm, auto iproj){
												 auto ist_spinor = ist%spinor_size;
												 double acc = 0.0;
												 auto pp = conj(proj[iproj][ilm][ist]);
												 for(int jlm = 0; jlm < nlm; jlm++) acc += real(pp*coe[iproj][ilm][jlm]*proj[iproj][jlm][ist]);
												 return occ[ist_spinor]*acc;
											 });
		
		if(reduce_states and phi.set_comm().size() > 1) {
			CALI_CXX_MARK_SCOPE("projector_all::energy::reduce_states");
			phi.set_comm().all_reduce_in_place_n(&en, 1, std::plus<>{});
		}

		return en;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename Phi, typename GPhi, typename Occupations, typename KPoint, typename Stress>
	void force_stress(Phi & phi, GPhi const & gphi, Occupations const & occupations, KPoint const & kpoint, gpu::array<vector3<double>, 1> & forces_non_local, Stress & stress) const {

		CALI_CXX_MARK_FUNCTION;
		
		namespace blas = boost::multi::blas;

		auto sphere_gphi = gather(gphi, kpoint);
		auto sphere_proj_phi = project(phi, kpoint);

		if(nprojs_ == 0 or max_sphere_size_ == 0) return;

		CALI_CXX_MARK_SCOPE("projector_force_sum");
		auto force = gpu::run(nprojs_, gpu::reduce(phi.local_set_size()), gpu::reduce(max_sphere_size_), zero<vector3<double, covariant>>(),
													[oc = begin(occupations), pphi = begin(sphere_proj_phi), gphi = begin(sphere_gphi), spinor_size = phi.local_spinor_set_size()] GPU_LAMBDA (auto iproj, auto ist, auto ip) {
														auto ist_spinor = ist%spinor_size;
														return -2.0*oc[ist_spinor]*real(pphi[iproj][ip][ist]*conj(gphi[iproj][ip][ist]));
													});
		
		for(auto iproj = 0; iproj < nprojs_; iproj++) {
			if(locally_empty_[iproj]) continue;
			
			forces_non_local[iatom_[iproj]] += phi.basis().volume_element()*phi.basis().cell().to_cartesian(force[iproj]);
		}
		
		auto lstress = gpu::run(gpu::reduce(nprojs_), gpu::reduce(phi.local_set_size()), gpu::reduce(max_sphere_size_), zero<Stress>(),
														[oc = begin(occupations), pphi = begin(sphere_proj_phi), gphi = begin(sphere_gphi), spinor_size = phi.local_spinor_set_size(),
														 pos = begin(positions_), cell = phi.basis().cell()]
														GPU_LAMBDA (auto iproj, auto ist, auto ip) {
															auto stress = zero<Stress>();
															
															auto grad_cart = cell.to_cartesian(gphi[iproj][ip][ist]);
															auto pos_cart = cell.to_cartesian(pos[iproj][ip]);
															
															for(auto alpha = 0; alpha < 3; alpha++) {
																for(auto beta = 0; beta < 3; beta++) {
																	stress[alpha][beta] = real(conj(grad_cart[alpha])*pos_cart[beta]*pphi[iproj][ip][ist]);
																}
															}
															
															auto ist_spinor = ist%spinor_size;
															return oc[ist_spinor]*stress;
														});
		stress += -4.0*phi.basis().volume_element()*lstress;

	}

	////////////////////////////////////////////////////////////////////////////////////////////
	// Calculates |cphi> += [Vnl, r] | phi>
	////////////////////////////////////////////////////////////////////////////////////////////	
	template <typename KpointType>
	void position_commutator(states::orbital_set<basis::real_space, complex> const & phi, states::orbital_set<basis::real_space, vector3<complex, covariant>> & cphi, KpointType const & kpoint) const {

		gpu::array<vector3<complex, contravariant>, 3> sphere_rphi_all({nprojs_, max_sphere_size_, phi.local_set_size()});
		gpu::array<vector3<complex, contravariant>, 3> rprojections_all({nprojs_, max_nlm_, phi.local_set_size()}, zero<vector3<complex, contravariant>>());

		{ CALI_CXX_MARK_SCOPE("position_commutator::gather");
				
			gpu::run(phi.local_set_size(), max_sphere_size_, nprojs_,
							 [srphi = begin(sphere_rphi_all), gr = begin(phi.hypercubic()), poi = begin(points_), pos = begin(positions_), kpoint] GPU_LAMBDA (auto ist, auto ipoint, auto iproj){
								 if(poi[iproj][ipoint][0] >= 0){
									 auto rr = static_cast<vector3<double, contravariant>>(pos[iproj][ipoint]);
									 auto phase = polar(1.0, dot(kpoint, rr));
									 srphi[iproj][ipoint][ist] = rr*phase*gr[poi[iproj][ipoint][0]][poi[iproj][ipoint][1]][poi[iproj][ipoint][2]][ist];
								 } else {
									 srphi[iproj][ipoint][ist][0] = complex(0.0, 0.0);
									 srphi[iproj][ipoint][ist][1] = complex(0.0, 0.0);
									 srphi[iproj][ipoint][ist][2] = complex(0.0, 0.0);
								 }
							 });
		}

	 	for(auto iproj = 0; iproj < nprojs_; iproj++){
			CALI_CXX_MARK_SCOPE("position_commutator_gemm_1");

			if(locally_empty_[iproj]) continue;
			
			namespace blas = boost::multi::blas;
			auto rpa = rprojections_all[iproj].template reinterpret_array_cast<complex>(3).rotated().flatted().unrotated();
			auto sra = sphere_rphi_all[iproj].template reinterpret_array_cast<complex>(3).rotated().flatted().unrotated();

			blas::real_doubled(rpa) = blas::gemm(phi.basis().volume_element(), matrices_[iproj], blas::real_doubled(sra));
		}

		multiply_by_coefficients(rprojections_all, coeff_);

		if(phi.basis().comm().size() > 1) {
			phi.basis().comm().all_reduce_in_place_n(raw_pointer_cast(rprojections_all.data_elements()), rprojections_all.num_elements(), std::plus<>{});
		}
		
		for(auto iproj = 0; iproj < nprojs_; iproj++) {
			CALI_CXX_MARK_SCOPE("position_commutator_gemm_2");

			if(locally_empty_[iproj]) continue;
			
			namespace blas = boost::multi::blas;
			auto rpa = rprojections_all[iproj].template reinterpret_array_cast<complex>(3).rotated().flatted().unrotated();
			auto sra = sphere_rphi_all[iproj].template reinterpret_array_cast<complex>(3).rotated().flatted().unrotated();
			blas::real_doubled(sra) = blas::gemm(1., blas::T(matrices_[iproj]), blas::real_doubled(rpa));			
		}

		auto sphere_phi_all = project(phi, kpoint);
		
		for(auto iproj = 0; iproj < nprojs_; iproj++){

			if(locally_empty_[iproj]) continue;
			
			gpu::run(phi.local_set_size(), max_sphere_size_,
							 [sgr = begin(sphere_phi_all), srphi = begin(sphere_rphi_all), gr = begin(cphi.hypercubic()), poi = begin(points_), iproj, pos = begin(positions_), kpoint, cell = phi.basis().cell()]
							 GPU_LAMBDA (auto ist, auto ipoint){
								 if(poi[iproj][ipoint][0] >= 0){
									 auto rr = static_cast<vector3<double, contravariant>>(pos[iproj][ipoint]);
									 auto phase = polar(1.0, -dot(kpoint, rr));
									 auto commutator = phase*cell.to_covariant(srphi[iproj][ipoint][ist] - rr*sgr[iproj][ipoint][ist]);
									 gpu::atomic::add(&gr[poi[iproj][ipoint][0]][poi[iproj][ipoint][1]][poi[iproj][ipoint][2]][ist], commutator);
								 }
							 });
		}
	}
  
};
  
}
}
#endif

#ifdef INQ_HAMILTONIAN_PROJECTOR_ALL_UNIT_TEST
#undef INQ_HAMILTONIAN_PROJECTOR_ALL_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	
}
#endif
