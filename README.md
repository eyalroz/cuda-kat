# cuda-kat: CUDA Kernel Author's Toolkit

An install-less, header-only library which is a loosely-coupled collection of **utility functions and classes** for writing **device-side CUDA code** (kernels and non-kernel functions). These utilities:

* Write templated device-side without constantly coming up against not-trivially-templatable bits.
* Use standard-library(-like) containers in device-side code (but not _have_ to use them).
* Not repeat ourselves as much (the [DRY principle](https://wiki.c2.com/?DontRepeatYourself)).
* Use less magic numbers.
* Make our device-side code less cryptic and idiosyncratic, with clearer naming and semantics.

... while not committing to any particular framework, library, paradigm or class hierarchy.


| Table of contents|
|:----------------|
| [Tools in the box](#what-is-in-the-box)<br>[Motivating example: The patterns that repeat!](#the-patterns-that-repeat) <br> [Questions? Suggestions? Found a bug?](#feedback)|

---

## <a name="what-is-in-the-box">The tools in the box</a>

The library has Doxygen documentation, available [here](https://codedocs.xyz/eyalroz/cuda-kat/). However - it is far from being complete. 

An alternative place to start is the table below. Since `cuda-kat`'s different facilities mostly correspond to different files (and not many such files), the table 
The various utility functions (and occasional other constructs) available in the `cuda-kat` package are accessed through the files in the following table. Each of these may be used independently of the others (although internally there are some dependencies); and no file imposes any particular abstraction or constraint on the rest of your code.

| Header(s)                        | Examples and description                                                |
|----------------------------------|-------------------------------------------------------------------------|
| [`grid_info.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/grid_info.cuh)         |`lane::id()`<br>`block::num_warps()`<br><br>Shorthands/mnemonics for information about positions and sizes of lanes, threads, warps and blocks within warps, blocks and grids (particularly for one-dimensional/linear grids). See also the [motivational section below](#the-patterns-that-repeat). |
| [`ptx.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/ptx.cuh)               | `ptx::is_in_shared_memory(const void* ptr)`<br>`ptx::bfind(uint32_t val)`<br>`ptx::special_registers::gridid()`<br><br>C++ bindings for PTX instructions which are not made available by CUDA itself. (Templatized versions of some of these appear in `builtins.cuh`.) |
| [`shared_memory.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/shared_memory.cuh)               | `shared_memory::proxy<T>()`<br><br>Access and size-determination gadgets for templated dynamic shared memory (which is not directly possible in CUDA!). |
| [`atomics.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/atomics.cuh)  | `atomic::increment(T* address)`<br>`atomic::apply_atomically<T, F>(T* address, F invokable)`<br><br>Uniform, templated, zero-overhead invocation of all CUDA-supported atomic operations; and compare-and-swap-based implementation for operations without atomic primitives on your GPU.  |
| [`builtins.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/builtins.cuh)  | `builtins::population_count()`<br>`builtins::maximum(T x, T y)`<br>`builtins::warp::some_lanes_satisfy(int condition)`<br><br> Uniform, templated, zero-overhead, non-cryptically-named invocation of all (non-atomic) operations involving a single PTX instructions (which aren't C++ language builtins like `+` or `-`). |
| [`shuffle.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/shuffle.cuh)  | Exchange arbitrary-type (and arbitrarily large) values using intra-warp shuffles. |
| [`math.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/math.cuh)  | Templated mathmetical functions - some `constexpr`, some using special GPU capabilities which cannot be exeucted at compile time even when data is available, and more efficient runtime implementation of compile-time-executable functions. |
| [`constexpr_math.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/constexpr_math.cuh)    | The part of `math.cuh` which can be `constexpr`-evaluated, i.e. executed at compile-time (but with `__host__ __device__` for good measure). |
| [`printing.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/streams/)          | `stringstream ss; ss << "easy as " << 123 << '!'; use_a_string(ss.c_str());`<br>`printfing_ostream cout; cout << "This text and number (" << 123.45 << ")\nget printfed on endl/flush" << flush;`<br><br>Stream functionality using dynamic memory allocation and/or CUDA's device-side `printf()`. Also supports automatic self-identification of the printer thread. |
| [`unaligned.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/unaligned.cuh)         | `read_unaligned_safe(const uint32_t* ptr)`<br><br>Read access to multi-byte values at non-naturally-aligned addresses (the GPU hardware _cannot_ directly read these). Caveat: UNTESTED. |
| [`sequence_ops/*.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/sequence_ops/)  | `block::collaborative::fill(data_start, data_end, 0xDEADBEEF)`<br>`warp::collaborative>(plus, result_ptr, data, )`<br> Grid, block and warp-level collaborative primitives of acti. Examples: Coalescing-friendly Visitation/traversal of a large index sequence; balloting, complex balloting operations, block-level broadcast, multisearch and more.  |
| [`collaboration/*.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/collaboration)  | Other grid, block and warp-level collaborative primitives (i.e. ones in which the threads in a grid, block, or warp have a common generic task to perform rather than per-thread task). Examples: Coalescing-friendly Visitation/traversal of a large index sequence; balloting, complex balloting operations, block-level broadcast, multisearch and more.  |
| [`c_standard_library/string.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/c_standard_library/string.cuh)  | Implementations of (almost) all functions in the C standard library's `<string.h>` header, to be executed at the single-thread level. This is neither quick nor efficient, but, you know, just in case you happen to need it.   |
| [`miscellany.cuh`](https://github.com/eyalroz/cuda-kat/blob/master/src/kat/on_device/miscellany.cuh)        | `swap(T& x, T&y)`<br><br>Functionality not otherwise cateogrized, but of sufficient worth to have available. |

For further details you can either explore the files themselves or read the Doxygen-genereated [official documentation](https://codedocs.xyz/eyalroz/cuda-kat/). Note The toolbox is not complete (especially the bottom items in the listing above), so expect some additions of files, and within files.


## <a name="the-patterns-that-repeat">Motivating example: The patterns that repeat! </a>

To illustrate the motivation for `cuda-kat` and the spirit in which it's written, let's focus on one kind of functionality: Referring to information about blocks, threads, warps, lanes and grids.

### <a name="an-obvious-example">You know you've written this before...</a>

As you write more CUDA kernels - for an application or a library - you likely begin to notice yourself repeating many small patterns of code. A popular one would be  computing the global id of a thread in a linear grid:
```
auto global_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
```
we've all been using that again and again, right? Many of us find ourselves writing a small utility function for this pattern. Perhaps it's something like:
```
auto global_thread_id() { return threadIdx.x + blockIdx.x * blockDim.x; }
```
In fact, some of those end up having written this same function several times, in several projects we've worked on. This is irksome, especially when you know that about every other CUDA library probably has this function, or uses this pattern, somewhere in its bowels. thrust does. cub does. cudf does. (moderngpu, weirdly, doesn't) and so on.

But here's the rub: When you do implement these convenience/utility functions, they're a part of something larger, which comes with its own assumptions, and dependencies, and abstractions. If you could only just share such small, low-level pieces of code among multiple projects without there being any "strings attached"...

Well, this is what `cuda-kat` is for.

### <a name="making-your-kernel-code-more-natural-language-like"> Say it with natural-language-like mnemonics</a>

The example of `global_thread_id()` is quite obvious in that it involves multiple operations and writing the function name is more terse than spelling out the computation. But naming this pattern is attractive just as much due to our giving of a **meaningful name** to the computation, regardless of its length. It's also an application of the  [DRY principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself); and it removes numerous repetitive calculations, which divert attention from the more meaningful computational work you're actually carrying out in your kernel.

But let's make that more concrete. Consider the following statements:
```
if (threadIdx.x == blockDim.x - 1) { do_stuff(); }
auto foo = threadIdx.x % warpSize;
auto bar = blockDim.x / warpSize;
```
when you write them, what you _really_ mean to be saying is:
```
if (this_thread_is_the_last_in_its_block()) { do_stuff(); }
auto foo = lane_id_within_warp();
auto bar = num_full_warps_in_grid_block();
```
and these, while longer to type, are clearer to the reader and less prone to typos.

### <a name="putting-things-in-order">... and put them all in order.</a>

Instead of a flat collection of rather long-named functions:
```
global_thread_id()
lane_id_within_warp()
this_thread_is_the_last_in_its_block()
num_full_warps_in_grid_block()
```
the library groups these (and many other related) functions into relevant namespaces. We thus have:
```
linear_grid::grid_info::thread::global_id()
grid_info::lane::id()
linear_grid::grid_info::thread::is_last_in_block()
linear_grid::grid_info::block::num_full_warps()
```
which is easier to browse through if you use auto-complete. The order comes at the expense of brevity... but we can alleviate this with an appropriate `namespace`. The above can then become simply:
```
namespace gi = kat::linear_grid::grid_info;
gi::thread::global_id()
gi::lane::id()
gi::thread::is_last_in_block()
gi::block::num_full_warps()
```
in your code. Now _this_ is how I want to write my kernels!

You will note, that most similar phrases you could come up with about positions and sizes within the grid - already have implementations. For example: "I can get the number of full warps, but now I want the number of warps, period"; well, just replace `num_full_warps()` with `num_warps()` and it's there: `linear_grid::grid_info::block::num_warps()` is available.

And as a final bonus - if you write a non-linear kernel, with blocks and grids having y and z dimensions other than 1 - you will only need to change your `namespace =` or `using` statements, to be able to write the same code and use 3-D implementations of these functions instead.


## <a name="feedback"> Questions? Suggestions? Found a bug?

* Have a question? There's a [FAQ](https://github.com/eyalroz/cuda-kat/wiki/FAQ) on the library's Wiki; please check it out first.
* Found a bug? A function/feature that's missing? A poor choice of design or of wording?-Please file an [issue](https://github.com/eyalroz/cuda-kat/issues/).
* Have a question that's _not_ in the FAQ? If you believe it's generally relevant, also [file an issue](https://github.com/eyalroz/cuda-kat/issues/), and clearly state that it's a question.
* Want to suggest significant additional functionality, which you believe would be of general interest? Either file an [issue](https://github.com/eyalroz/cuda-kat/issues/) or [write me](mailto:euyalroz@technion.ac.il) about it.
* Pull Requests [are welcome](https://github.com/eyalroz/cuda-kat/pulls), but it's better to work things out by talking to me before you invest a lot of work in preparing a PR.
