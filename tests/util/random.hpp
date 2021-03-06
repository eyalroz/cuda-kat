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
#include <type_traits>
#include <unordered_set>

namespace util {

namespace random {


extern std::random_device device;  // Note this is a callable object.
extern std::default_random_engine engine;

using result_t = decltype(engine)::result_type;
using seed_t   = result_t;

template <typename T>
using uniform_distribution = std::conditional_t<
	std::is_floating_point<T>::value,
	std::uniform_real_distribution<T>,
	std::uniform_int_distribution<T>
>;



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

template <typename Inserter, typename Size, typename Distribution, typename Engine = std::default_random_engine>
constexpr inline void insertion_generate_n(
	Inserter inserter,
	Size count,
	Distribution&& distribution,
	Engine& engine = util::random::engine)
{
	insertion_generate_n(inserter, count, distribution, engine);
}

template <typename RandomAccessIterator, typename Size, typename Engine = std::default_random_engine>
constexpr inline std::unordered_set<typename std::iterator_traits<RandomAccessIterator>::value_type>
sample_subset(
	RandomAccessIterator begin,
	RandomAccessIterator end,
	Size subset_size,
	Engine& engine = util::random::engine)
{
	std::unordered_set<typename std::iterator_traits<RandomAccessIterator>::value_type> sampled_subset{};
	std::uniform_int_distribution<Size> distribution {0, (end - begin) - 1};
	while(sampled_subset.size() < subset_size) {
		auto sampled_element_index = util::random::sample_from(distribution, engine);
		sampled_subset.insert(*(begin + sampled_element_index));
	}
	return sampled_subset;
}

template <typename RandomAccessIterator, typename Size, typename Engine = std::default_random_engine>
constexpr inline std::unordered_set<typename std::iterator_traits<RandomAccessIterator>::value_type>
sample_subset(
	RandomAccessIterator begin,
	Size domain_length,
	Size subset_size,
	Engine& engine = util::random::engine)
{
	if (domain_length < subset_size) { throw std::invalid_argument("Can't sample a subset larger than the domain"); }
	std::unordered_set<typename std::iterator_traits<RandomAccessIterator>::value_type> sampled_subset{};
	if (domain_length == 0) {
		if (subset_size == 0) { return sampled_subset; }
		throw std::invalid_argument("Can't sample a subset larger than the domain");
	}
	std::uniform_int_distribution<Size> distribution {0, domain_length - 1};
	// TODO: If we need to sample more than half the domain, sample the elements _outside_ the set instead.
	while(sampled_subset.size() < subset_size) {
		auto sampled_element_index = util::random::sample_from(distribution, engine);
		sampled_subset.insert(*(begin + sampled_element_index));
	}
	return sampled_subset;
}

template <typename Size, typename Engine = std::default_random_engine>
constexpr inline std::unordered_set<Size>
sample_index_subset(
	Size domain_length,
	Size subset_size,
	Engine& engine = util::random::engine)
{
	if (domain_length < subset_size) { throw std::invalid_argument("Can't sample a subset larger than the domain"); }
	std::unordered_set<Size> sampled_subset{};
	if (domain_length == 0) {
		if (subset_size == 0) { return sampled_subset; }
		throw std::invalid_argument("Can't sample a subset larger than the domain");
	}
	std::uniform_int_distribution<Size> distribution {0, domain_length - 1};
	// TODO: If we need to sample more than half the domain, sample the elements _outside_ the set instead.
	while(sampled_subset.size() < subset_size) {
		sampled_subset.insert(util::random::sample_from(distribution, engine));
	}
	return sampled_subset;
}

} // namespace random
} // namespace util

#endif /* CUDA_KAT_TEST_UTILITIES_RANDOM_H_ */

