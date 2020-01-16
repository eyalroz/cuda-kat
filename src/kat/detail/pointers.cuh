#pragma once
#ifndef CUDA_KAT_POINTERS_CUH_
#define CUDA_KAT_POINTERS_CUH_

#include <type_traits>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace detail {

static constexpr auto obj_ptr_size { sizeof(void *)     };
static constexpr auto fun_ptr_size { sizeof(void (*)()) };
//auto dat_mem_ptr_size = sizeof(generic_dat_mem_ptr_t);
//auto mem_fun_size = sizeof(generic_mem_fun_ptr_t);

//auto max_ptr_size = std::max({ sizeof(generic_obj_ptr_t), sizeof(generic_fun_ptr_t), sizeof(generic_dat_mem_ptr_t), sizeof(generic_mem_fun_ptr_t) });
//auto max_ptr_align = std::max({ alignof(generic_obj_ptr_t), alignof(generic_fun_ptr_t), alignof(generic_dat_mem_ptr_t), alignof(generic_mem_fun_ptr_t) });

static constexpr auto max_ptr_size { (obj_ptr_size > fun_ptr_size) ? obj_ptr_size : fun_ptr_size };

static_assert(max_ptr_size == sizeof(uint64_t), "Unexpected maximum pointer size");

using address_t = uint64_t;

//KAT_FHD address_t address_as_number (address_t       address) { return address; }
template <typename T>
constexpr KAT_FHD address_t address_as_number (const T*  address) noexcept { return reinterpret_cast<address_t>(address); }
template <typename T>
constexpr KAT_FHD T*        address_as_pointer(address_t address) noexcept { return reinterpret_cast<T*>(address); }

template <typename T1, typename T2>
KAT_FHD std::ptrdiff_t address_difference(T1* p1, T2* p2)
{
	return address_as_number(p1) - address_as_number(p2);
}


// TODO: Code duplication with math.cuh
template <typename I>
constexpr KAT_FHD bool is_power_of_2(I val) { return (val & (val-1)) == 0; }

template <typename T>
constexpr KAT_FHD address_t misalignment_extent(address_t address) noexcept
{
	static_assert(is_power_of_2(sizeof(T)),"Invalid type for alignment");
	constexpr address_t mask = sizeof(T) - 1; // utilizing the fact that it's a power of 2
	return address & mask;
}

/**
 * Computes the number of bytes by which a pointer is misaligned.
 *
 * @tparam T  The pointer-to element type; its size must be a power of 2.
 *
 * @param ptr The possibly-misaligned pointer
 * @return the minimum number of bytes which, if deducted from ptr, produces
 * a T-aligned pointer
 */
template <typename T>
constexpr KAT_FHD address_t misalignment_extent(const T* ptr) noexcept
{
	return misalignment_extent<T>(address_as_number(ptr));
}

template <typename T>
constexpr KAT_FHD bool is_aligned(const T* ptr) noexcept
{
	return misalignment_extent(ptr) == 0;
}

template <typename T>
constexpr KAT_FHD bool is_aligned(address_t address) noexcept
{
	return misalignment_extent<T>(address) == 0;
}

template <typename T>
constexpr KAT_FHD address_t align_down(address_t address) noexcept
{
	return (address - misalignment_extent<T>(address));
}

/**
 * @tparam T a type whose size is a power of 2 (and thus has natural alignment)
 * @param A possibly-unaligned pointer
 * @return A pointer to the closest aligned T in memory upto and including @p ptr
 */
template <typename AlignBy, typename T>
KAT_FHD AlignBy* align_down(T* ptr) noexcept
{
	// Note: The compiler _should_ optimize out the inefficiency of using
	// misalignment_extent rather than just applying a mask once.
	address_t address = address_as_number(ptr);
	auto aligned_down_addr = align_down<AlignBy>(address);
	return (AlignBy*) address_as_pointer<AlignBy*>(aligned_down_addr);
}

template <typename T>
KAT_FHD T* align_down(T* ptr) noexcept
{
	return const_cast<T*>(align_down<T>(reinterpret_cast<const T*>(ptr)));
}


} // namespace detail
} // namespace kat

#include <kat/detail/execution_space_specifiers.hpp>

#endif // CUDA_KAT_POINTERS_CUH_
