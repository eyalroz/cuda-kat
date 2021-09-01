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

find_package(CUDAToolkit REQUIRED)
find_package(cuda-api-wrappers 0.4.3 REQUIRED)

include(CMakeDependentOption)
set(KEEP_PTX FALSE CACHE BOOL "Keep kernel PTX files for build targets")


#############
##  Tests  ##
#############

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})


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
	array
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
	tuple
	sequence_ops
	ranges
)

foreach(TEST_TARGET ${tests})
	add_executable(${TEST_TARGET} "${TEST_TARGET}.cu")
	target_compile_options(${TEST_TARGET} PRIVATE "--expt-relaxed-constexpr")
	target_compile_options(${TEST_TARGET} PRIVATE "--expt-extended-lambda")
	target_link_libraries(${TEST_TARGET} PRIVATE cuda-kat cuda-api-wrappers::runtime-api doctest  ${CUDA_LIBRARIES} test_utils)
	# I don't see why the following line should even be necessary. Depending on the libraries should be enough to get us their include dirs!
	target_include_directories(${TEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)

	set_target_properties(${TEST_TARGET}
		PROPERTIES
		CXX_STANDARD 14
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS NO
		CUDA_STANDARD 14
		CUDA_STANDARD_REQUIRED YES
		CUDA_EXTENSIONS NO
	)
	#	set_target_properties(${TEST_TARGET} PROPERTIES COMPILE_FLAGS "-std=c++14" )
	add_test(
		NAME ${PROJECT_NAME}.${TEST_TARGET}
		COMMAND ${TEST_TARGET} ${TEST_RUNNER_PARAMS}
	)
	if (KEEP_PTX)
	        target_compile_options(${TEST_TARGET} PRIVATE "--keep")
	endif()
endforeach(TEST_TARGET)

# TODO: Something about code coverage perhaps?
