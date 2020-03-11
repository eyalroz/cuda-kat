cmake_minimum_required(VERSION 3.8.2)


###############
##  Modules  ##
###############

# Standard CMake modules

# Custom modules

include(DocTest)

################
##  Packages  ##
################

find_package(CUDA 8.0 REQUIRED)
find_package(cuda-api-wrappers 0.3.0 REQUIRED)
find_package(cuda-nvtx REQUIRED) # Actually, it's sort-of required by cuda-api-wrappers

include(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(BUILD_PRINTING_RELATED_TESTS "Build (strf-based) printing-related tests" ON "BUILD_TESTS" ON)
set(KEEP_PTX FALSE CACHE BOOL "Keep kernel PTX files for build targets")

# This overcomes some linking issues I've encountered... I'm sure there's a better solution
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_NVTX_LIBRARY} ${CUDA_cudadevrt_LIBRARY})


#############
##  Tests  ##
#############

cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS_TMP Auto)
set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS_TMP} CACHE STRING "CUDA gencode parameters")
string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS_STR}")

#add_test(
#    # Use some per-module/project prefix so that it is easier to run only tests for this module
#    NAME ${PROJECT_NAME}.failtest
#    COMMAND failtest ${TEST_RUNNER_PARAMS}
#)
##target_set_warnings(${TEST_MAIN} ENABLE ALL AS_ERROR ALL DISABLE Annoying) # Set warnings (if needed).
#set_tests_properties(
#    ${PROJECT_NAME}.failtest
#    PROPERTIES
#        WILL_FAIL TRUE # We expect this test to fail
#)

add_library(test_utils util/random.cu)
set_target_properties(
	test_utils
	PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED YES
	CXX_EXTENSIONS NO
)

set(tests
	shared_memory
	math
	shuffle
	atomics
	constexpr_math
	time
	c_string
	span
	miscellany
	builtins
	grid_collaboration
	block_collaboration
	warp_collaboration
)

if (BUILD_PRINTING_RELATED_TESTS)
	list(APPEND tests printing)
endif()

foreach(TEST_TARGET ${tests})
	add_executable(${TEST_TARGET} "${TEST_TARGET}.cu")
	target_compile_options(${TEST_TARGET} PRIVATE "--expt-relaxed-constexpr")
	target_compile_options(${TEST_TARGET} PRIVATE "--expt-extended-lambda")
	target_link_libraries(${TEST_TARGET} PRIVATE cuda-kat cuda-api-wrappers::cuda-api-wrappers doctest  ${CUDA_LIBRARIES} test_utils)
	# I don't see why the following line should even be necessary. Depending on the libraries should be enough to get us their include dirs!
	target_include_directories(${TEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)

	set_target_properties(
		${TEST_TARGET}
		PROPERTIES
		CXX_STANDARD 14
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS NO
	)
	add_test(
		NAME ${PROJECT_NAME}.${TEST_TARGET}
		COMMAND ${TEST_TARGET} ${TEST_RUNNER_PARAMS}
	)
	if (KEEP_PTX)
	        target_compile_options(${TEST_TARGET} PRIVATE "--keep")
	endif()
endforeach(TEST_TARGET)

if (BUILD_PRINTING_RELATED_TESTS)
	target_link_libraries(printing PRIVATE strf::strf-header-only)
	target_compile_options(printing PRIVATE --ptxas-options --suppress-stack-size-warning)
endif()

# TODO: Something about code coverage perhaps?
