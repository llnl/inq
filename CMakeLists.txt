# -*- indent-tabs-mode: t -*-
# Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.21)
message("CMake version ${CMAKE_VERSION}")

enable_language(CXX)
include(GNUInstallDirs)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

project(inq VERSION 0.1)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

if(NOT DEFINED CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

#always optimize the code
if(ENABLE_CUDA)
  set(CMAKE_CXX_FLAGS  "-O3 ${CMAKE_CXX_FLAGS} -fno-lto")  # in newer versions of Ubuntu, LTO is on by default, which is incompatible with NVCC
else()
  set(CMAKE_CXX_FLAGS  "-O3 ${CMAKE_CXX_FLAGS}")
endif()

message("CMake build type: ${CMAKE_BUILD_TYPE}")

set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "")
set(FETCHCONTENT_QUIET FALSE)
Include(FetchContent)

add_library(inq INTERFACE)
target_include_directories(inq
    INTERFACE
     $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
     $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
     $<INSTALL_INTERFACE:src>
)

target_compile_options(inq INTERFACE
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
  --expt-relaxed-constexpr --extended-lambda --Werror=cross-execution-space-call -Xcudafe=--display_error_number -Xcudafe=--diag_error=incompatible_assignment_operands -Xcudafe=--diag_error=returning_ptr_to_local_variable -Xcudafe=--diag_error=subscript_out_of_range -Xcudafe=--diag_error=used_before_set -Xcudafe=--diag_error=undefined_preproc_id -Xcudafe=--diag_error=implicit_func_decl -Xcudafe=--diag_error=implicit_return_from_non_void_function -Xcudafe=--diag_error=missing_type_specifier
  >
)

enable_testing()
include(CTest)

target_compile_features(inq INTERFACE cxx_std_17)

if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 11.5 REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  #   include(FindCUDA/select_compute_arch)
  #   CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  #   string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  #   string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  #   string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  #   SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  #   # set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
  #   # set_property(TARGET inq PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    SET(CMAKE_CUDA_ARCHITECTURES native)
  endif()


  # string(APPEND CMAKE_CUDA_FLAGS " --forward-unknown-to-host-linker ${ARCH_FLAGS} -std=c++17 --expt-relaxed-constexpr --extended-lambda --Werror=cross-execution-space-call -Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\"")

  target_link_libraries(inq INTERFACE CUDA::cublas CUDA::cufft CUDA::cudart CUDA::cusolver)

	if(ENABLE_NCCL)
		find_package(NCCL REQUIRED)
		target_include_directories(inq INTERFACE ${NCCL_INCLUDE_DIRS})
		target_link_libraries(inq INTERFACE ${NCCL_LIBRARIES})
  endif()

	set(ENABLE_GPU TRUE)    
	set(GPU_LANGUAGE CUDA)
endif()

if(ENABLE_HIP)
  enable_language(HIP)

	find_package(hip REQUIRED)
  find_package(hipFFT REQUIRED)
  find_package(hipblas REQUIRED)

	target_link_libraries(inq INTERFACE hip::host)
  target_link_libraries(inq INTERFACE hip::hipfft)
  target_link_libraries(inq INTERFACE roc::hipblas)

	find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")     # On ROCm rocThrust requires rocPRIM
	find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")

	target_link_libraries(inq INTERFACE roc::rocthrust)

  set(CMAKE_HIP_FLAGS "-fgpu-rdc ${CMAKE_HIP_FLAGS}")
	
	set(ENABLE_GPU TRUE)
	set(GPU_LANGUAGE HIP)
endif()

link_libraries(m)

enable_language(Fortran)

include(FortranCInterface)
FortranCInterface_VERIFY(CXX)
FortranCInterface_HEADER(FC.h MACRO_NAMESPACE "FC_")

find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(inq INTERFACE MPI::MPI_CXX)

message(STATUS "MPI Executable for running programs:" ${MPIEXEC_EXECUTABLE})
message(STATUS "MPI number of processors detected on the host system: " ${MPIEXEC_MAX_NUMPROCS})
include(CMakePrintHelpers)
cmake_print_properties(TARGETS MPI::MPI_CXX PROPERTIES INTERFACE_LINK_LIBRARIES INTERFACE_INCLUDE_DIRECTORIES)

#Check for MPI_Isendrecv_replace that sometimes MPI doesn't have
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH})
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
include(CheckCXXSymbolExists)
check_cxx_symbol_exists(MPI_Isendrecv_replace mpi.h HAVE_MPI_ISENDRECV_REPLACE)

#FFTW has to go before blas to avoid unscrupulous (i.e. MKL) implementations that include FFTW and don't implement the full FFTW interface
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW fftw3 IMPORTED_TARGET)
if(FFTW_FOUND)
  target_link_libraries(inq INTERFACE PkgConfig::FFTW)
else()  # some systems do not have fftw3 registered in pkg-config, in this case it falls back to good old -lfftw3 which can fail later
  if(NOT DEFINED FFTW_LIBRARIES)
    message(WARNING "FFTW not detected by pkg-config, using fftw3 or define -DFFTW_LIBRARIES (e.g. =-lfftw3) and -DFFTW_INCLUDE_DIRS (e.g. =/usr/include)")
    target_link_libraries(inq INTERFACE fftw3)
    target_include_directories(inq INTERFACE ${FFTW_INCLUDE_DIRS})
  else()
    target_link_libraries(inq INTERFACE ${FFTW_LIBRARIES})
    target_include_directories(inq INTERFACE ${FFTW_INCLUDE_DIRS})
  endif()
  check_cxx_symbol_exists(fftw_plan_dft_1d fftw3.h FFTW3_CHECKED)
  if(FFTW_FOUND)
  else()
    message(FATAL_ERROR "FFTW not found")
  endif()
endif()

# Multi
#set(multi_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external_libs/multi/boost/include)

add_subdirectory(${PROJECT_SOURCE_DIR}/external_libs/multi)
add_subdirectory(${PROJECT_SOURCE_DIR}/external_libs/multi/include/boost/multi/adaptors/blas)

target_link_libraries(inq INTERFACE multi)

# find_package(BLAS REQUIRED)
# target_link_libraries(inq INTERFACE ${BLAS_LIBRARIES})

# ESSL doesn't implement dgelss_, needed in the code, may need to override set(BLA_VENDOR Generic)
find_package(LAPACK REQUIRED)
target_link_libraries(inq INTERFACE ${LAPACK_LIBRARIES})

#GPU::RUN
add_subdirectory(${PROJECT_SOURCE_DIR}/external_libs/gpurun)
set(gpurun_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external_libs/gpurun/include)
target_include_directories(inq INTERFACE ${PROJECT_SOURCE_DIR}/external_libs/gpurun/include)

#SHARMONIC
add_subdirectory(external_libs/sharmonic)
target_include_directories(inq INTERFACE ${PROJECT_SOURCE_DIR}/external_libs/sharmonic/include ${PROJECT_BINARY_DIR}/external_libs/sharmonic)

#LIBXC
message("\nCONFIGURING LIBXC\n")
set(DISABLE_FXC ON)
set(BUILD_TESTING OFF)
add_subdirectory(external_libs/libxc EXCLUDE_FROM_ALL)
message("\nDONE CONFIGURING LIBXC\n")
set(Libxc_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external_libs/libxc/src/ ${PROJECT_BINARY_DIR}/external_libs/libxc ${PROJECT_BINARY_DIR}/external_libs/libxc/gen_funcidx)

target_include_directories(inq INTERFACE ${Libxc_INCLUDE_DIRS})
target_link_libraries(inq INTERFACE ${Libxc_LIBRARIES} xc)

########################
# LIBPAW INTERFACE
########################

message("\nCONFIGURING LIBPAW INTERFACE\n")
add_subdirectory(external_libs/libpaw_interface)
target_include_directories(inq INTERFACE ${PROJECT_SOURCE_DIR}/external_libs/libpaw_interface/)
target_include_directories(inq INTERFACE ${PROJECT_BINARY_DIR}/external_libs/libpaw_interface/)
message("\nDONE CONFIGURING LIBPAW INTERFACE\n")
set(libpaw_interface_include_dirs ${PROJECT_SOURCE_DIR}/external_libs/libpaw_interface)

target_link_libraries(inq INTERFACE libpaw_interface)


#PSEUDOPOD
add_subdirectory(external_libs/pseudopod)
target_include_directories(inq INTERFACE ${PROJECT_BINARY_DIR}/external_libs/pseudopod)


# MPI3
# set(bmpi3_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external_libs/mpi3/include)

add_subdirectory(${PROJECT_SOURCE_DIR}/external_libs/mpi3)
target_link_libraries(inq INTERFACE bmpi3)

########################
# SPDLOG
########################

message(STATUS "Downloading spdlog")
set(SPDLOG_BUILD_PIC "YES")
set(SPDLOG_NO_THREAD_ID "YES")
set(SPDLOG_NO_TLS "YES")

FetchContent_Declare(
  spdlog
	GIT_REPOSITORY https://github.com/gabime/spdlog.git
	GIT_TAG        v1.11.0
  GIT_PROGRESS   TRUE
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(spdlog)

set(spdlog_INCLUDE_DIRS ${spdlog_SOURCE_DIR}/include/)
target_include_directories(inq INTERFACE ${spdlog_INCLUDE_DIRS})
target_link_libraries(inq INTERFACE spdlog)

########################
# SPGLIB
########################
message(STATUS "Downloading spglib")

set(CMAKE_BUILD_TYPE_BACKUP ${MAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE Release)

FetchContent_Declare(
  spglib
	GIT_REPOSITORY https://gitlab.com/npneq/spglib.git
	GIT_TAG        f04dadd3175a60279f86780630045d3b3f64b8aa
  GIT_PROGRESS   TRUE
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(spglib)

set_property(DIRECTORY ${spglib_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)

set(CMAKE_BUILD_TYPE ${MAKE_BUILD_TYPE_BACKUP})

set(spglib_INCLUDE_DIRS ${spglib_SOURCE_DIR}/src/)

target_include_directories(inq
  INTERFACE
  ${spglib_INCLUDE_DIRS}
)
target_link_libraries(inq INTERFACE symspg_static)
target_link_libraries(inq
   INTERFACE
   ${spglib_LIBRARIES}
)

########################
# GEMMI
########################

target_include_directories(inq INTERFACE ${PROJECT_SOURCE_DIR}/external_libs/gemmi/include)

########################
# ZLIB
########################

find_package(ZLIB REQUIRED)
target_link_libraries(inq INTERFACE ZLIB::ZLIB)

########################
# BOOST
########################

find_package(Boost REQUIRED COMPONENTS serialization system)
message(STATUS "Boost actual version found " ${Boost_VERSION})
if(Boost_VERSION LESS_EQUAL "1.53.0")  # 105300
	message(STATUS "Applying option BOOST_NO_AUTO_PTR to patch for old Boost")
	add_compile_options(-DBOOST_NO_AUTO_PTR)
endif()

target_include_directories(inq INTERFACE Boost::serialization Boost::system)
target_link_libraries(inq INTERFACE Boost::serialization Boost::system)
target_compile_definitions(inq INTERFACE "BOOST_PP_VARIADICS=1")  # NVCC needs BOOST_PP_VARIADICS to compile some Boost headers

########################
# PYTHON
########################

find_package(Python COMPONENTS Interpreter Development)
set(PYBIND11_PYTHON_VERSION ${Python_VERSION})  # Force Pybind11 into the same version as Python package https://stackoverflow.com/a/62838072/225186

add_subdirectory(python)



#######################
# GENERATE inq_config.h
######################

include_directories(${PROJECT_BINARY_DIR})

target_include_directories(inq INTERFACE ${PROJECT_BINARY_DIR})
target_include_directories(inq INTERFACE ${PROJECT_SOURCE_DIR}/external_libs)

configure_file(inq_config.h.in inq_config.h)
configure_file(scripts/run_test.sh.in scripts/run_test.sh @ONLY)

install(DIRECTORY share/ DESTINATION share/inq)

# this makes this library CMake's FetchContent friendly https://www.foonathan.net/2022/06/cmake-fetchcontent/
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  return()
endif()

message(STATUS "Downloading Catch2 v3p3p2 for inq")
FetchContent_Declare(
	Catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG        v3.3.2
  GIT_PROGRESS   TRUE
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
#link_libraries(Catch2::Catch2WithMain)

#####################
# BUILD DIRECTORIES
#####################

set(INQ_PATHS
  "INQ_SHARE_PATH=set:${PROJECT_SOURCE_DIR}/share"
  "PSEUDOPOD_SHARE_PATH=set:${PROJECT_SOURCE_DIR}/external_libs/pseudopod/share"
  "PYTHONPATH=path_list_prepend:${PROJECT_BINARY_DIR}/python"
  "PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/python"
  "PATH=path_list_prepend:${PROJECT_BINARY_DIR}/src"
)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(speed_tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)
