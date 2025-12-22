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
  target_link_libraries(${TEST_EXE} PRIVATE Catch2::Catch2 inq)
  add_test(NAME inq::tests::${TEST_EXE} COMMAND ${PROJECT_BINARY_DIR}/scripts/run_test.sh ${CMAKE_CURRENT_BINARY_DIR}/${TEST_EXE})
	set_property(TEST inq::tests::${TEST_EXE} PROPERTY ENVIRONMENT_MODIFICATION "${INQ_PATHS}")
  if(ENABLE_GPU)
    set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE ${GPU_LANGUAGE})
  endif()
endforeach()

#PYTHON TESTS

file(GLOB PYTHON_TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)

foreach(TEST_FILE ${PYTHON_TESTS})
  add_test(NAME inq::tests::${TEST_FILE} COMMAND ${PROJECT_BINARY_DIR}/scripts/run_test.sh ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE})
  set_property(TEST inq::tests::${TEST_FILE} PROPERTY ENVIRONMENT_MODIFICATION "${INQ_PATHS}")
endforeach()

#BASH TESTS

file(GLOB COMMAND_LINE_TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.sh)

foreach(TEST_FILE ${COMMAND_LINE_TESTS})
  add_test(NAME inq::tests::${TEST_FILE} COMMAND ${PROJECT_SOURCE_DIR}/scripts/run_shell_test.sh ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE})
  set_property(TEST inq::tests::${TEST_FILE} PROPERTY ENVIRONMENT_MODIFICATION "${INQ_PATHS}")
endforeach()
