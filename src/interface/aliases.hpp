/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__ALIASES
#define INQ__INTERFACE__ALIASES

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <unordered_map>

namespace inq {
namespace interface {
auto aliases() {
	using namespace std::string_literals;
	
	return std::unordered_map<std::string, std::string>{
		{ "atomicnumber"s,             "atomic-number"s            },
		{ "calculator"s,               "calc"s                     },
		{ "clean"s,                    "clear"s                    },
		{ "densitycutoff"s,            "density-cutoff"s           },
		{ "exactexchange"s,            "exact-exchange"s           },
		{ "extraelectrons"s,           "extra-electrons"s          },
		{ "extrastates"s,              "extra-states"s             },
		{ "filename"s,                 "file"s                     },
		{ "force"s,                    "forces"s                   },		
		{ "fraction"s,                 "fractional"s               },
		{ "freq"s,                     "frequency"s                },
		{ "functionals"s,              "functional"s               },
		{	"groundstate"s,              "ground-state"s             },
		{	"gs"s,                       "ground-state"s             },
		{	"hartreefock"s,              "hartree-fock"s             },
		{	"kpoint"s,                   "kpoints"s                  },
		{	"kpoints"s,                  "kpoints"s                  },
		{	"k-point"s,                  "kpoints"s                  },
		{	"k-points"s,                 "kpoints"s                  },
		{	"listsets"s,                 "list-sets"s                },		
		{ "maxsteps"s,                 "max-steps"s                },
		{ "mix"s,                      "mixing"s                   },
		{ "noncollinear"s,             "non-collinear"s            },
		{ "noninteracting"s,           "non-interacting"s          },
		{ "nonlocal"s,                 "non-local"s                },
		{ "observable"s,               "observables"s              },
		{ "perturbation"s,             "perturbations"s            },
		{ "propagationtime"s,          "propagation-time"s         },
		{ "pseudo-potential"s,         "pseudo"s                   },
		{ "pseudopotential"s,          "pseudo"s                   },
		{ "pseudoset"s,                "pseudo-set"s               },
		{ "realtime"s,                 "real-time"s                },
		{ "rt"s,                       "real-time"s                },
		{ "result"s,                   "results"s                  },
		{ "results-ground-state"s,     "results ground-state"s     },
		{ "resultsgroundstate"s,       "results ground-state"s     },
		{ "resultsground-state"s,      "results ground-state"s     },
		{ "results-groundstate"s,      "results ground-state"s     },
		{ "resultsgs"s,                "results ground-state"s     },
		{ "results-gs"s,               "results ground-state"s     },
		{ "results-real-time"s,        "results real-time"s        },
		{ "resultsrealtime"s,          "results real-time"s        },
		{ "resultsreal-time"s,         "results real-time"s        },
		{ "results-realtime"s,         "results real-time"s        },
		{ "resultsrt"s,                "results real-time"s        },
		{ "results-rt"s,               "results real-time"s        },
		{ "grid-shifted"s,             "shifted-grid"s             },
		{ "shiftedgrid"s,              "shifted-grid"s             },
		{ "timestep"s,                 "time-step"s                },
		{ "totaltime"s,                "total-time"s               },
		{ "totalsteps"s,               "total-steps"s              },
		{ "tol"s,                      "tolerance"s                },
		{ "utils"s,                    "util"s                     }
	};
}
}
}

#endif

#ifdef INQ_INTERFACE_ALIASES_UNIT_TEST
#undef INQ_INTERFACE_ALIASES_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
