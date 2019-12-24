#pragma once
#ifndef CUDA_KAT_TEST_UTILITIES_RANDOM_H_
#define CUDA_KAT_TEST_UTILITIES_RANDOM_H_

/************************************************************
 *
 * Simplistic and non-thread-safe random number generation
 * convenience utility - based on the C++ standard library.
 *
 * If you need to do something serious with random numbers,
 * dont use this; if you just want a bunch of random-looking
 * numbers quick & dirty, do use it.
 *
 ************************************************************/

#include <random>
#include <algorithm>
#include <iterator>

namespace util {

namespace random {


extern std::random_device device;  // Note this is a callable object.
extern std::default_random_engine engine;

using result_t = decltype(engine)::result_type;
using seed_t   = result_t;


/*
// TODO: Does the distribution object actually remain constant? I wonder.
// Should I return an rvalue reference?
template <typename Distribution>
inline typename Distribution::result_type sample_from(Distribution& distribution) {
	return distribution(engine);
}
*/

template <typename Distribution, typename Engine = std::default_random_engine>
inline typename Distribution::result_type sample_from(
	Distribution&  distribution,
	Engine&        engine = util::random::engine)
{
	return distribution(engine);
}

inline void seed(const seed_t& seed_value)
{
	engine.seed(seed_value);
}

/* In your code, do something like:

	const int rangeMin = 1;
	const int rangeMax = 10;
	std::uniform_int_distribution<uint32_t> distribution(rangeMin, rangeMax);
	// util::random::seed(std::time(0)); // seed with the current time
	auto a = util::random::sample_from(distribution);
	cout << "A random integer between " << rangeMin << "and " << " for you: "
		  << util::random::sample_from(distribution) << '\n';

*/

// Some more examples of distributions:
//std::uniform_int_distribution<uint32_t> uint_dist;         // by default range [0, MAX]
//std::uniform_int_distribution<uint32_t> uint_dist10(0,10); // range [0,10]
//std::normal_distribution<double> normal_dist(mean, stddeviation);  // N(mean, stddeviation)

template <typename ForwardIt, typename Distribution, typename Engine = std::default_random_engine>
constexpr inline void generate(
	ForwardIt first,
	ForwardIt last,
	Distribution& distribution,
	Engine& engine = util::random::engine)
{
	// If we could rely on having C++17, we could generate in parallel...
	std::generate(first, last, [&distribution, &engine]() {
		return static_cast<typename std::iterator_traits<ForwardIt>::value_type>(sample_from(distribution, engine));
	});
}

template <typename ForwardIt, typename Size, typename Distribution, typename Engine = std::default_random_engine>
constexpr inline void generate_n(
	ForwardIt first,
	Size count,
	Distribution& distribution,
	Engine& engine = util::random::engine)
{
//	static_assert(is_iterator<ForwardIt>::value == true, "The 'first' parameter is not of an iterator type");
	// If we could rely on having C++17, we could generate in parallel...
	return generate(first, first + count, distribution, engine);
}

template <typename Inserter, typename Size, typename Distribution, typename Engine = std::default_random_engine>
constexpr inline void insertion_generate_n(
	Inserter inserter,
	Size count,
	Distribution& distribution,
	Engine& engine = util::random::engine)
{
	for(size_t i = 0; i < count; i++) {
		*(inserter++) = sample_from(distribution, engine);
	}
}


} // namespace random
} // namespace util

#endif /* CUDA_KAT_TEST_UTILITIES_RANDOM_H_ */

