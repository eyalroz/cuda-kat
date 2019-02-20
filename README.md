# cuda-kat: CUDA Kernel Author's Tools

An install-less, header-only library of small, self-contained **pieces of utility code** for writing device-side CUDA functions and kernels.


| Table of contents|
|:----------------|
| [Tools in the box](#what-is-in-the-box)<br>[Motivating example: The patterns that repeat!](#the-patterns-that-repeat) <br> [Bugs, suggestions, feedback](#feedback)|

---

## <a name="what-is-in-the-box">The tools in the box</a>

The various utility functions (and occasional other constructs) available in the `cuda-kat` package are accessed through the files in the following table. Each of these may be used independently of the others (although internally there are some dependencies); and no file imposes any particular abstraction or constraint on the rest of your code.

| Header(s)                        | Description                                                             |
|----------------------------------|-------------------------------------------------------------------------|
| [`grid_info.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/grid_info.cuh)         | Shorthands/mnemonics for information about positions and sizes of lanes, threads, warps and blocks within warps, blocks and grids (particularly for one-dimensional/linear grids). See also the [motivational section below](#the-patterns-that-repeat). |
| [`ptx.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/ptx.cuh)               | C++ bindings for PTX instructions which are not made available by CUDA itself: Special registers, [`bfind`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind), [`prmt`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt) and so on. Used by code in `wrappers/` |
| [`shared_memory.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/shared_memory.cuh)               | Mainly - a proxy for accessing templated dynamic shared memory which is not directly possible in CUDA; shared memory size getters; and code to divvy up shared memory among the thread blocks. |
| [`wrappers/atomics.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/wrappers/atomics.cuh)  | A uniform, templated zero-overhead way to perform all CUDA-supported atomic operations, and generic `apply_atomically()` for not-directly-supported unary and binary functions.  |
| [`wrappers/builtins.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/wrappers/builtins.cuh)  | A uniform, templated zero-overhead way to execute any (non-atomic) operations which involves a single PTX instructions and are not C++ language builtins - similarly to the host-side compiler builtins: bit operations; warp balloting, conjunction and disjunction, intra-warp shuffles and lane mask manipulation. |
| [`wrappers/shuffle.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/wrappers/shuffle.cuh)  | Exchange arbitrary-type (and arbitrarily large) values using intra-warp shuffles. |
| [`constexpr_math.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/constexpr_math.cuh)    | Mathmetical functions which may be executed at compile-time --- as some are only available as non-`constexpr` in the C++ standard library; and others are not available at all (bit-resolution functions specialized for power-of-2 arguments. |
| [`math.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/math.cuh)  | Mathmetical functions which cannot be exeucted at compile time even when data is available, and more efficient runtime implementation of compile-time-executable functions. |
| [`printing.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/printing.cuh)          | More elaborate functionality on top of CUDA's device-side `printf()`, such as: Printing once in every warp, block or the entire grid; printing bit-resolution values; printing with the thread/warp/block identifiers prefixed. |
| [`unaligned.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/unaligned.cuh)         | Access multi-byte values at unaligned non-naturally-aligned addressed (something the GPU hardware itself will _not_ do directly). Caveat: Mostly Untested. |
| [`miscellany.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/miscellany.cuh)        | A proxy for shared memory for use in templated code (avoiding name clashes); an alternative `memcpy()`; a swap function; etc. |
| [`primitives/warp.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/primitives/warp.cuh)  | Warp-level computational primitives (i.e. ones in which the lanes in a warp have a common task to perform rather than per-lane tasks). Examples: Copying, reduction, complex balloting operations (which require more than merely a single builtin).  |
| [`primitives/block.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/primitives/block.cuh)  | Block-level computational primitives (i.e. ones in which the threads in a block have a common task to perform rather than per-thread or per-warp tasks). Examples: copying, reduction, certain patters of sharing data among threads in different warps.   |
| [`primitives/grid.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/primitives/grid.cuh)  | Patterns of covering a sequence of data with a grid, given a per-element function; Consolidation primitives for per-warp and per-block data into global (per-grid) data.   |

For further details you can either explore the files themselves or read the Doxygen-genereated [official documentation](https://codedocs.xyz/eyalroz/cuda-kat/). Note The toolbox is not complete (especially the bottom items in the listing above), so expect some additions of files, and within files.


## <a name="the-patterns-that-repeat">Motivating example: The patterns that repeat! </a>

To illustrate the motivation for `cuda-kat` and the spirit in which it's written, let's focus on one kind of functionality: Referring to information about blocks, threads, warps, lanes and grids.

### <a name="an-obvious-example">You know you've written this before...</a>

As you write more CUDA kernels - for an application or a library - you likely begin to notice yourself repeating many small patterns of code. A popular one would be  computing the global index of a thread in a linear grid:
```
auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
```
we've all been using that again and again, right? Many of us find ourselves writing a small utility function for this pattern. Perhaps it's something like:
```
auto global_thread_index() { return threadIdx.x + blockIdx.x * blockDim.x; }
```
In fact, some of those end up having written this same function several times, in several projects we've worked on. This is irksome, especially when you know that about every other CUDA library probably has this function, or uses this pattern, somewhere in its bowels. thrust does. cub does. cudf does. (moderngpu, weirdly, doesn't) and so on.

But here's the rub: When you do implement these convenience/utility functions, they're a part of something larger, which comes with its own assumptions, and dependencies, and abstractions. If you could only just share such small, low-level pieces of code among multiple projects without there being any "strings attached"...

Well, this is what `cuda-kat` is for.

### <a name="making-your-kernel-code-more-natural-language-like"> Say it with natural-language-like mnemonics</a>

The example of `global_thread_index()` is quite obvious in that it involves multiple operations and writing the function name is more terse than spelling out the computation. But naming this pattern is attractive just as much due to our giving of a **meaningful name** to the computation, regardless of its length. It's also an application of the  [DRY principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself); and it removes numerous repetitive calculations, which divert attention from the more meaningful computational work you're actually carrying out in your kernel.

But let's make that more concrete. Consider the following statements:
```
if (threadIdx.x == blockDim.x - 1) { do_stuff(); }
auto foo = threadIdx.x % warpSize;
auto bar = blockDim.x / warpSize;
```
when you write them, what you _really_ mean to be saying is:
```
if (this_thread_is_the_last_in_its_block()) { do_stuff(); }
auto foo = lane_index_within_warp();
auto bar = num_full_warps_in_grid_block();
```
and these, while longer to type, are clearer to the reader and less prone to typos.

### <a name="putting-things-in-order">... and put them all in order.</a>

Instead of a flat collection of rather long-named functions:
```
global_thread_index()
lane_index_within_warp()
this_thread_is_the_last_in_its_block()
num_full_warps_in_grid_block()
```
the library groups these (and many other related) functions into relevant namespaces. We thus have:
```
grid_info::linear::thread::global_index()
grid_info::lane::index()
grid_info::linear::thread::is_last_in_block()
grid_info::linear::block::num_full_warps()
```
which is easier to browse through if you use auto-complete. The order comes at the expense of brevity... but we can fix this by issuing the appropriate `namespace` or `using namespace` commands. The above can then become simply:
```
thread::global_index()
lane::index()
thread::is_last_in_block()
block::num_full_warps()
```
in your code. Now _this_ is how I want to write my kernels!

You will note, that most similar phrases you could come up with about positions and sizes within the grid - already have implementations. For example: "I can get the number of full warps, but now I want the number of warps, period"; well, just replace `num_full_warps()` with `num_warps()` and it's there: `grid_info::linear::block::num_warps()` is available.

## <a name="feedback"> Bugs, suggestions, feedback

* If you've found a bug; a function that's missing; or a poor design/wording choice I've made - please file an [issue](https://github.com/eyalroz/cuda-kat/issues/).
* If you want to suggest significant additional functionality, which you believe would be of general interest - either file an [issue](https://github.com/eyalroz/cuda-kat/issues/) or [write me](mailto:euyalroz@technion.ac.il) about it.
* Pull Requests [are welcome](https://github.com/eyalroz/cuda-kat/pulls), but it's better to work things out by talking to me before you invest a lot of work in preparing a PR.
