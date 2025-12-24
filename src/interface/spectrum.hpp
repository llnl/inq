/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__SPECTRUM
#define INQ__INTERFACE__SPECTRUM

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <ground_state/initial_guess.hpp>
#include <real_time/propagate.hpp>
#include <real_time/results.hpp>

namespace inq {
namespace interface {

struct {

	constexpr auto name() const {
		return "spectrum";
	}

	constexpr auto one_line() const {
		return "Calculates different types of spectra based on calculation results.";
	}
	
	constexpr auto help() const {
		return R""""(

The 'spectrum' command
==================

This command is used to calculate spectra of different types using the
results of calculations.

These are the options available:

-  Shell:  `spectrum file <filename>`
   Python: `spectrum.file(filename)`

   Calculates an absorption spectra based on a dipole series. The
   spectrum data is printed to the terminal as two data columns, the
   first one is the frequency/energy and the second one is the
   spectrum for all input columns..

   Shell examples:  `inq spectrum file data.txt`
   Python examples: `pinq.spectrum.file("data.txt")`


)"""";
	}

  static void print(gpu::array<complex, 2> const & freq_series) {
    std::cout << std::scientific;
    for(auto ifreq = 0; ifreq < (~freq_series).size(); ifreq++){
      std::cout << real(freq_series[0][ifreq]);
      for(int icol = 1; icol < freq_series.size(); icol++) std::cout << '\t' << real(freq_series[icol][ifreq]) << '\t' << imag(freq_series[icol][ifreq]);
      std::cout << '\n';
    }
  }
  
  template <typename Stream>
  static auto from_stream(Stream & inp) {

    std::vector<double> values;
    
    auto ncolumns = 0;
    while(true) {
      auto line = std::string();
      getline(inp, line);
      if(inp.eof()) break;
      if(line[0] == '#') continue;

      //      std::cout << line << std::endl;

      if(ncolumns == 0) { //count the columns
        std::istringstream iss(line + " ");
        
        std::string data_item;
        while(iss) {
          iss >> data_item;
          if(iss.eof()) break;
          ncolumns++;
        }
      }
      
      std::istringstream iss(line + " ");
      
      double val;
      for(int icol = 0; icol < ncolumns; icol++) {
        iss >> val;
        values.push_back(val);
      }

    }

    assert(values.size()%ncolumns == 0);

    auto ntime = (long) (values.size())/ncolumns;
    
    gpu::array<double, 1> time(ntime);
    gpu::array<double, 2> time_series({ncolumns - 1, ntime});
    

    for(auto itime = 0l; itime < ntime; itime++){
      time[itime] = values[itime*ncolumns];
      for(auto icol = 1; icol < ncolumns; icol++) time_series[icol - 1][itime] = values[itime*ncolumns + icol];
    }

    using namespace magnitude;
    auto maxw = 20.0_eV;
    auto dw = 0.01_eV;

    long nfreq = maxw/dw + 1;

    gpu::array<complex, 2> freq_series({ncolumns, nfreq});

    for(auto ifreq = 0; ifreq < nfreq; ifreq++) freq_series[0][ifreq] = ifreq*dw.in_atomic_units();
    
    for(auto icol = 0; icol < ncolumns - 1; icol++){
      freq_series[1 + icol] = observables::spectrum(maxw, dw, time, time_series[icol]);
    }

    return freq_series;
  }

  static auto file(std::string const & filename) {
    std::ifstream inp(filename);    
    return from_stream(inp);
  }

  static auto stdin() {
    return from_stream(std::cin);
  }
  
  template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {

		if(args.size() == 2 and args[0] == "file") {
      auto sp = file(args[1]);
      print(sp);
			actions::normal_exit();
		}

		if(args.size() == 1 and args[0] == "stdin") {
      auto sp = stdin();
      print(sp);
			actions::normal_exit();
		}
    
  }
	
#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;
 
		auto sub = module.def_submodule(name(), help());

    sub.def("file", [](std::string const & filename) {
      auto arr = file(filename);
			
			py::array_t<complex, py::array::c_style> py_arr({arr.size(), (~arr).size()});
			
			auto acc = py_arr.mutable_unchecked();
			
			for (py::ssize_t ii = 0; ii < acc.shape(0); ii++) {
				for (py::ssize_t jj = 0; jj < acc.shape(1); jj++) {
					acc(ii, jj) = arr[ii][jj];
				}
			}
		
			return py_arr;
    });
    
	}
#endif

} const spectrum;

}
}
#endif

#ifdef INQ_INTERFACE_SPECTRUM_UNIT_TEST
#undef INQ_INTERFACE_SPECTRUM_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
