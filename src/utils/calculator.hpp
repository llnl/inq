/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__UTILS__CALCULATOR
#define INQ__UTILS__CALCULATOR

// Copyright (C) 2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/spirit/home/x3.hpp>
#include <string_view>

namespace inq {
namespace utils {
namespace calculator {

namespace x3 = boost::spirit::x3;

namespace parsing {

static x3::rule<struct _expr, double> expr("expr");
static x3::rule<struct _expo, double> expo("expo");
static x3::rule<struct _term, double> term("term");
static x3::rule<struct _factor, double> factor("factor");

using x3::_val, x3::_attr;

// NVCC (EDG?) doesn't tolerate lambdas defined inside operator[]
constexpr auto identity_fn  = [](auto& ctx) {_val(ctx) = _attr(ctx);};

constexpr auto posit_fn     = [](auto& ctx) {_val(ctx) = + _attr(ctx);};
constexpr auto negate_fn    = [](auto& ctx) {_val(ctx) = - _attr(ctx);};

constexpr auto sqrt_fn      = [](auto& ctx) {_val(ctx) = std::sqrt(_attr(ctx));};
constexpr auto exp_fn       = [](auto& ctx) {_val(ctx) = std::exp (_attr(ctx));};
constexpr auto cos_fn       = [](auto& ctx) {_val(ctx) = std::cos (_attr(ctx));};
constexpr auto sin_fn       = [](auto& ctx) {_val(ctx) = std::sin (_attr(ctx));};
constexpr auto log_fn       = [](auto& ctx) {_val(ctx) = std::log (_attr(ctx));};

constexpr auto pow_fn       = [](auto& ctx) {
	using boost::fusion::at_c;
	_val(ctx) = std::pow(at_c<0>(_attr(ctx)), at_c<1>(_attr(ctx)));
};

constexpr auto pi_fn        = [](auto& ctx) {_val(ctx) = M_PI;};
constexpr auto e_fn         = [](auto& ctx) {_val(ctx) = M_E ;};

constexpr auto pow_acc_fn   = [](auto& ctx) {_val(ctx) = std::pow(_val(ctx), _attr(ctx));};

constexpr auto multiplies_acc_fn = [](auto& ctx) {_val(ctx) *= _attr(ctx);};
constexpr auto divides_acc_fn    = [](auto& ctx) {_val(ctx) /= _attr(ctx);};

constexpr auto plus_acc_fn  = [](auto& ctx) {_val(ctx) += _attr(ctx);};
constexpr auto minus_acc_fn = [](auto& ctx) {_val(ctx) -= _attr(ctx);};

// clang-format off
auto const unit = x3::rule<struct _unit, double>("unit") =
	  (    x3::double_    )[identity_fn]
	| (("(" >> term) > ")")[identity_fn]

	| ("-" >> term)[negate_fn  ]
	| ("+" >> term)[posit_fn   ]

	| (("sqrt(" >> term) > ')')[sqrt_fn]
	| (( "exp(" >> term) > ')')[exp_fn ]
	| (( "cos(" >> term) > ')')[cos_fn ]
	| (( "sin(" >> term) > ')')[sin_fn ]
	| (( "log(" >> term) > ')')[log_fn ]

	| (("pow(" >> term) > ',' > term > ')')[pow_fn]

	| (x3::lit("pi"))[pi_fn  ]
	| (x3::lit("e" ))[e_fn   ]
;

auto const expo_def = unit[identity_fn] >> *('^' > term [pow_acc_fn]);

auto const factor_def = expo [identity_fn] >> *(('*' > term[multiplies_acc_fn]) | ('/' > term[divides_acc_fn   ]));

auto const term_def =
	factor[identity_fn] >> *(('+' > term[plus_acc_fn ]) | ('-' > term[minus_acc_fn]))
;
// clang-format on

static auto expr_def = x3::skip(x3::space)[x3::eps > term > x3::eoi];

BOOST_SPIRIT_DEFINE(expr, term, factor, expo)  // This may need a compile definition -DBOOST_PP_VARIADICS=1 in NVCC
} // namespace parsing

static auto eval(std::string_view text) try {
	double result;
	if (!x3::parse(text.begin(), text.end(), calculator::parsing::expr, result)) {
		throw std::runtime_error("cannot parse expression\n\"" + std::string{text} + "\"");
	}
	return result;
} catch (x3::expectation_failure<std::string_view::const_iterator> const &err) {
	throw std::runtime_error(
		"error parsing expression \n\"" + std::string(text) + "\"\n"
		+ std::string(err.where() - text.begin() + 1, ' ') +
		"^---- here\n...while applying parsing rule named \"" + err.which() + "\""
	);
}

} // end namespace calculator
} // end namespace utils
} // end namespace inq

#endif

#ifdef INQ_UTILS_CALCULATOR_UNIT_TEST
#undef INQ_UTILS_CALCULATOR_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;

	using inq::utils::calculator::eval;

	// to interactively check results try here https://godbolt.org/z/7nqT6335v
	CHECK(eval("1.0") != 2.0);

	CHECK(eval("1.0") == 1.0);
	CHECK(eval("2.0") == 2.0);

	CHECK(eval("1.0 + 2.0") == 3.0);
	CHECK(eval("1.0 - 2.0") == 1.0 - 2.0);

	CHECK(eval("1.0 + 2.0e3") == 1.0 + 2.0e3);

	CHECK(eval("2.0 * 3.0") == 6.0);

	CHECK(eval("4 + (2 * 3)") == 10.0);
	CHECK(eval("4 + (2 * 3)") == 10.0);

	CHECK(eval("pi") == M_PI );
	CHECK(eval("e") == M_E );
	CHECK(eval("e + 1.0") == M_E + 1.0);

	CHECK(eval("sqrt(2.0)") == std::sqrt(2.0) );
	CHECK(eval("sqrt(2.0 + 1.0)") == std::sqrt(2.0 + 1.0) );

	CHECK(eval("sqrt(3)*1.42") == std::sqrt(3.0)*1.42 );
	CHECK(eval("3.2^2") == std::pow(3.2, 2.0) );
	CHECK(eval("1.1 + (2.2 + 3.3)") == 1.1 + 2.2 + 3.3 );
	CHECK(eval("sin(pi/4) + cos(pi/2)") == std::sin(M_PI/4.0) + cos(M_PI/2.0) );

	CHECK(eval("exp(-2)") == std::exp(-2.0) );
	CHECK(eval("log(e^3.0)") == std::log(std::exp(3.0)) );
	CHECK(eval(" log(e^3.0)") == std::log(std::exp(3.0)) );
	CHECK(eval(" log(e^3.0) ") == std::log(std::exp(3.0)) );
	CHECK(eval(" log(e ^ 3.0) ") == std::log(std::exp(3.0)) );

	CHECK(eval("pow(e, 3.0) ") == std::pow(M_E, 3.0) );
	CHECK(eval("pow(2.0, 3.0) ") == std::pow(2.0, 3.0) );

	CHECK(eval("(1.0)") == 1.0);
	CHECK(eval("(1.0 + 2.0)") == 1.0 + 2.0);

	CHECK(eval("-(1.0 + 2.0)") == -(1.0 + 2.0) );
	CHECK(eval("+(1.0 + 2.0)") == +(1.0 + 2.0) );
	CHECK(eval("+sqrt(1.0 + 2.0)") == +std::sqrt(1.0 + 2.0) );
	CHECK(eval("-sqrt(1.0 + 2.0)") == -std::sqrt(1.0 + 2.0) );

	CHECK(eval("-1.0") == -1.0 );

	CHECK_NOTHROW(eval("1.0 + 2.0"));

	CHECK_THROWS(eval("poop (1.0 + 2.0)"));
	CHECK_THROWS(eval("(1.0 + poop 2.0)"));
	CHECK_THROWS(eval("(1.0 + 2.0) poop"));
}
#endif
