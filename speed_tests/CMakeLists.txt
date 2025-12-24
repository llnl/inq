# Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

include_directories(.)
include_directories(../external_libs)
include_directories(../src)

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(TEST_FILE ${TEST_SRCS})
  get_filename_component(TEST_EXE ${TEST_FILE} NAME_WE)
  add_executable(${TEST_EXE} ${TEST_FILE})
  target_include_directories(${TEST_EXE} PRIVATE ${catch2_SOURCE_DIR}/src)
  target_link_libraries(${TEST_EXE} PRIVATE unit_tests_main Catch2::Catch2 inq)
  if(ENABLE_GPU)
    set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE ${GPU_LANGUAGE})
  endif()
endforeach()
