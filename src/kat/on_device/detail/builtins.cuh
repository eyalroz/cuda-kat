/*
 * These are templated wrappers for actual primitives. No function
 * here performs any calculation (except a type cast in some rare
 * cases, or masking an inopportune interface;
 * non-builtin operations belong in other files.
 */
#ifndef CUDA_KAT_ON_DEVICE_BUILTINS_DETAIL_CUH_
#define CUDA_KAT_ON_DEVICE_BUILTINS_DETAIL_CUH_

#include <kat/on_device/ptx.cuh>

#if ! __CUDA_ARCH__ >= 300
#error "This code can only target devices of compute capability 3.0 or higher."
#endif


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

// TODO: Consider using "sized types" (from cstdint) more. It makes more sense when hardware is involved,
// although - CUDA-provided functions don't really do that and just make implicit assumptions about
// the sizes of int, unsigned, long etc.

namespace builtins {

template <> KAT_FD  int                multiplication_high_bits<int               >( int x,                int y)                { return __mulhi(x, y);    }
template <> KAT_FD  unsigned           multiplication_high_bits<unsigned          >( unsigned x,           unsigned y)           { return __umulhi(x, y);   }
template <> KAT_FD  long long          multiplication_high_bits<long long         >( long long x,          long long y)          { return __mul64hi(x, y);  }
template <> KAT_FD  unsigned long long multiplication_high_bits<unsigned long long>( unsigned long long x, unsigned long long y) { return __umul64hi(x, y); }

template <> KAT_FD  float  divide<float >(float dividend, float divisor)   { return fdividef(dividend, divisor); }
template <> KAT_FD  double divide<double>(double dividend, double divisor) { return fdividef(dividend, divisor); }

// TODO: Does this really translate into a single instruction? I'm worried the casting might incur more than a single one.
template <typename I> KAT_FD int population_count(I x)
{
	static_assert(std::is_integral<I>::value, "Only integral types are supported");
	static_assert(sizeof(I) <= sizeof(unsigned long long), "Unexpectedly large type");

	using native_popc_type =
		typename std::conditional<
			sizeof(I) <= sizeof(unsigned),
			unsigned,
			unsigned long long
		>::type;
	return population_count<native_popc_type>(static_cast<native_popc_type>(x));
}

template <> KAT_FD int population_count<unsigned>(unsigned x)                     { return __popc(x);   }
template <> KAT_FD int population_count<unsigned long long>(unsigned long long x) { return __popcll(x); }

template <> KAT_FD int      sum_with_absolute_difference<int     >(int x,      int y,      int addend)      { return __sad (x, y, addend); }
template <> KAT_FD unsigned sum_with_absolute_difference<unsigned>(unsigned x, unsigned y, unsigned addend) { return __usad(x, y, addend); }

template <> KAT_FD int                absolute_value<int                >(int x)                { return abs(x);   }
template <> KAT_FD long               absolute_value<long               >(long x)               { return labs(x);  }
template <> KAT_FD long long          absolute_value<long long          >(long long x)          { return llabs(x); }
template <> KAT_FD float              absolute_value<float              >(float x)              { return fabsf(x); }
template <> KAT_FD double             absolute_value<double             >(double x)             { return fabs(x);  }
template <> KAT_FD unsigned char      absolute_value<unsigned char      >(unsigned char x)      { return x;        }
template <> KAT_FD unsigned short     absolute_value<unsigned short     >(unsigned short x)     { return x;        }
template <> KAT_FD unsigned           absolute_value<unsigned           >(unsigned x)           { return x;        }
template <> KAT_FD unsigned long      absolute_value<unsigned long      >(unsigned long x)      { return x;        }
template <> KAT_FD unsigned long long absolute_value<unsigned long long >(unsigned long long x) { return x;        }

template <> KAT_FD int                bit_reverse<int               >(int x)                { return __brev(x);   }
template <> KAT_FD unsigned           bit_reverse<unsigned          >(unsigned x)           { return __brev(x);   }
template <> KAT_FD long long          bit_reverse<long long         >(long long x)          { return __brevll(x); }
template <> KAT_FD unsigned long long bit_reverse<unsigned long long>(unsigned long long x) { return __brevll(x); }


namespace special_registers {

// TODO: Should we really specify the types here, or just DRY and forward the registers' types using auto?
KAT_FD unsigned           lane_index()                     { return ptx::special_registers::laneid();            }
KAT_FD unsigned           symmetric_multiprocessor_index() { return ptx::special_registers::smid();              }
KAT_FD unsigned long long grid_index()                     { return ptx::special_registers::gridid();            }
KAT_FD unsigned int       dynamic_shared_memory_size()     { return ptx::special_registers::dynamic_smem_size(); }
KAT_FD unsigned int       total_shared_memory_size()       { return ptx::special_registers::total_smem_size();   }

} // namespace special_registers

namespace bit_field {

template <> KAT_FD uint32_t extract(uint32_t bit_field, unsigned start_pos, unsigned num_bits)
{
	return ptx::bfe(bit_field, start_pos, num_bits);
}

template <> KAT_FD uint64_t extract(uint64_t bit_field, unsigned start_pos, unsigned num_bits)
{
	return ptx::bfe(bit_field, start_pos, num_bits);
}

template <> KAT_FD uint32_t insert(uint32_t bits_to_insert, uint32_t existing_bit_field, uint32_t start_pos, uint32_t num_bits)
{
	return ptx::bfi(bits_to_insert, existing_bit_field, start_pos, num_bits);
}

template <> KAT_FD uint64_t insert(uint64_t bits_to_insert, uint64_t existing_bit_field, uint32_t start_pos, uint32_t num_bits)
{
	return ptx::bfi(bits_to_insert, existing_bit_field, start_pos, num_bits);
}

} // namespace bit_field

template <> KAT_FD unsigned select_bytes(unsigned x, unsigned y, unsigned byte_selector) { return ptx::prmt(x, y, byte_selector); }

#if __CUDA_ARCH__ >= 320

template <
	typename T,
	funnel_shift_amount_resolution_mode_t AmountResolutionMode
>
KAT_FD native_word_t funnel_shift(
	native_word_t  low_word,
	native_word_t  high_word,
	native_word_t  shift_amount)
{
	return (AmountResolutionMode == detail::funnel_shift_amount_resolution_mode_t::take_lower_bits) ?
		__funnelshift_l(low_word, high_word, shift_amount) :
		__funnelshift_lc(low_word, high_word, shift_amount);
}

#endif // __CUDA_ARCH__ >= 320

template <bool Signed, bool Rounded> KAT_FD
typename std::conditional<Signed, int, unsigned>::type average(
	typename std::conditional<Signed, int, unsigned>::type x,
	typename std::conditional<Signed, int, unsigned>::type y);

template <> KAT_FD unsigned average<false, false>(unsigned x, unsigned y) { return __uhadd(x,y);  }
template <> KAT_FD int      average<true,  false>(int      x, int y     ) { return __hadd(x,y);   }
template <> KAT_FD unsigned average<false, true >(unsigned x, unsigned y) { return __urhadd(x,y); }
template <> KAT_FD int      average<true,  true >(int      x, int y     ) { return __rhadd(x,y);  }


namespace warp {

#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD lane_mask_t ballot           (int condition) { return __ballot(condition); }
KAT_FD int         all_lanes_satisfy(int condition) { return __all(condition);    }
KAT_FD int         any_lanes_satisfy(int condition) { return __any(condition);    }
#else
KAT_FD lane_mask_t ballot   (int condition, lane_mask_t lane_mask) { return __ballot_sync(lane_mask, condition); }
KAT_FD int all_lanes_satisfy(int condition, lane_mask_t lane_mask) { return __all_sync(lane_mask, condition);    }
KAT_FD int any_lanes_satisfy(int condition, lane_mask_t lane_mask) { return __any_sync(lane_mask, condition);    }
KAT_FD int all_lanes_agree  (int condition, lane_mask_t lane_mask) { return __uni_sync(lane_mask, condition);    }

#endif

#if (__CUDACC_VER_MAJOR__ >= 9)

template <typename T>
KAT_FD bool is_uniform_across_lanes (T value, lane_mask_t lane_mask)
{
	// __match_all_sync has a weirdly redundant signature and semantics!
	// Consult the CUDA Programming guide v9 or later for more details
	int dummy;
	__match_all_sync(lane_mask, value, &dummy);
	return (dummy != 0);
}

template <typename T>
KAT_FD bool is_uniform_across_warp(T value) { return is_uniform_across_lanes<T>(full_warp_mask, value); }

template <typename T> KAT_FD lane_mask_t matching_lanes(lane_mask_t lanes, T value) { return __match_any_sync(lanes, value); }
template <typename T> KAT_FD lane_mask_t matching_lanes(T value) { return __match_any_sync(full_warp_mask, value); }
#endif


namespace mask_of_lanes {

KAT_FD lane_mask_t preceding()           { return ptx::special_registers::lanemask_lt(); }
KAT_FD lane_mask_t preceding_and_self()  { return ptx::special_registers::lanemask_le(); }
KAT_FD lane_mask_t self()                { return ptx::special_registers::lanemask_eq(); }
KAT_FD lane_mask_t succeeding_and_self() { return ptx::special_registers::lanemask_ge(); }
KAT_FD lane_mask_t succeeding()          { return ptx::special_registers::lanemask_gt(); }

#if (__CUDACC_VER_MAJOR__ >= 9)
template <typename T>
KAT_FD lane_mask_t lanes_matching_values(T value, lane_mask_t lane_mask)
{
	return __match_any_sync(lane_mask, value);
}
#endif // __CUDACC_VER_MAJOR__ >= 9


} // namespace mask_of_lanes

namespace shuffle {

#if (__CUDACC_VER_MAJOR__ < 9)
template <typename T> KAT_FD T arbitrary(T x, int source_lane, int width)        { return __shfl(x, source_lane, width);   }
template <typename T> KAT_FD T down(T x, unsigned delta, int width)              { return __shfl_down(x, delta, width);    }
template <typename T> KAT_FD T up(T x, unsigned delta, int width)                { return __shfl_up(x, delta, width);      }
template <typename T> KAT_FD T xor_(T x, int xoring_mask_for_lane_id, int width) { return __shfl_xor(x, xoring_mask_for_lane_id, width); }
	// we have to use xor_ here since xor is a reserved word
#else
template <typename T> KAT_FD T arbitrary( T x, int source_lane, int width, lane_mask_t participating_lanes)
{
	return __shfl_sync(participating_lanes, x, source_lane, width);
}
template <typename T> KAT_FD T down(T x, unsigned delta, int width, lane_mask_t participating_lanes)
{
	return __shfl_down_sync(participating_lanes, x, delta, width);
}
template <typename T> KAT_FD T up(T x, unsigned delta, int width, lane_mask_t participating_lanes)
{
	return __shfl_up_sync(participating_lanes, x, delta, width);
}
template <typename T> KAT_FD T xor_(T x, int lane_id_xoring_mask, int width, lane_mask_t participating_lanes)
{
	return __shfl_xor_sync(participating_lanes, x, lane_id_xoring_mask, width);
}
#endif

} // namespace shuffle


} // namespace warp

template <>  KAT_FD uint32_t find_last_non_sign_bit<int               >(int x)                  { return ptx::bfind(x);            }
template <>  KAT_FD uint32_t find_last_non_sign_bit<unsigned          >(unsigned x)             { return ptx::bfind(x);            }
template <>  KAT_FD uint32_t find_last_non_sign_bit<long              >(long x)                 { return ptx::bfind((int64_t) x);  }
template <>  KAT_FD uint32_t find_last_non_sign_bit<unsigned long     >(unsigned long x)        { return ptx::bfind((uint64_t) x); }
template <>  KAT_FD uint32_t find_last_non_sign_bit<long long         >(long long x)            { return ptx::bfind((int64_t) x);  }
template <>  KAT_FD uint32_t find_last_non_sign_bit<unsigned long long>(unsigned long long  x)  { return ptx::bfind((uint64_t) x); }

template <typename T>
KAT_FD T load_global_with_non_coherent_cache(const T* ptr)  { return ptx::ldg(ptr); }

// Note: We can't generalize clz to an arbitrary type, without subtracting the size difference from the result of the builtin clz instruction.

template <> KAT_FD int count_leading_zeros<int               >(int x)                { return __clz(x);   }
template <> KAT_FD int count_leading_zeros<unsigned          >(unsigned x)           { return __clz(x);   }
template <> KAT_FD int count_leading_zeros<long long         >(long long x)          { return __clzll(x); }
template <> KAT_FD int count_leading_zeros<unsigned long long>(unsigned long long x) { return __clzll(x); }

template <> KAT_FD int                 minimum<int               >(int x, int y)                               { return min(x,y);    }
template <> KAT_FD unsigned int        minimum<unsigned          >(unsigned int x, unsigned int y)             { return umin(x,y);   }
template <> KAT_FD long                minimum<long              >(long x, long y)                             { return llmin(x,y);  }
template <> KAT_FD unsigned long       minimum<unsigned long     >(unsigned long x, unsigned long y)           { return ullmin(x,y); }
template <> KAT_FD long long           minimum<long long         >(long long x, long long y)                   { return llmin(x,y);  }
template <> KAT_FD unsigned long long  minimum<unsigned long long>(unsigned long long x, unsigned long long y) { return ullmin(x,y); }
template <> KAT_FD float               minimum<float             >(float x, float y)                           { return fminf(x,y);  }
template <> KAT_FD double              minimum<double            >(double x, double y)                         { return fmin(x,y);   }

template <> KAT_FD int                 maximum<int               >(int x, int y)                               { return max(x,y);    }
template <> KAT_FD unsigned int        maximum<unsigned          >(unsigned int x, unsigned int y)             { return umax(x,y);   }
template <> KAT_FD long                maximum<long              >(long x, long y)                             { return llmax(x,y);  }
template <> KAT_FD unsigned long       maximum<unsigned long     >(unsigned long x, unsigned long y)           { return ullmax(x,y); }
template <> KAT_FD long long           maximum<long long         >(long long x, long long y)                   { return llmax(x,y);  }
template <> KAT_FD unsigned long long  maximum<unsigned long long>(unsigned long long x, unsigned long long y) { return ullmax(x,y); }
template <> KAT_FD float               maximum<float             >(float x, float y)                           { return fmaxf(x,y);  }
template <> KAT_FD double              maximum<double            >(double x, double y)                         { return fmax(x,y);   }

} // namespace builtins
} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_BUILTINS_DETAIL_CUH_
