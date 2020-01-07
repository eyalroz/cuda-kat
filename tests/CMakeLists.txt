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
find_package(cuda-api-wrappers REQUIRED)
find_package(cuda-nvtx REQUIRED) # Actually, it's sort-of required by cuda-api-wrappers

include(CMakeDependentOption)
CMAKE_DEPENDENT_OPTION(BUILD_PRINTING_RELATED_TESTS "Build (strf-based) printing-related tests" ON "BUILD_TESTS" ON)

# This overcomes some linking issues I've encountered... I'm sure there's a better solution
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_NVTX_LIBRARY} ${CUDA_cudadevrt_LIBRARY})


#############
##  Tests  ##
#############

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

# TODO: We are currently extremely hacky about the libraries and include dependencies of the tests;
# that definitely needs to be improved. In particular, we should just depend on cuda-api-wrappers which
# should pull in everything else.

# TODO 2: We're using an extra object target for each test to ensure compilation of .ptx files (the _obj
# targets). We should try to avoid those.

add_library(test_utils util/random.cu)
set_target_properties(
	test_utils
	PROPERTIES 
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED YES 
	CXX_EXTENSIONS NO
)

set(tests shared_memory math shuffle atomics constexpr_math)

if (BUILD_PRINTING_RELATED_TESTS)
	list(APPEND tests printing)
endif()

foreach(TEST_TARGET ${tests})
	add_executable(${TEST_TARGET} "${TEST_TARGET}.cu")
	target_compile_options(${TEST_TARGET} PRIVATE "--expt-relaxed-constexpr")
	target_link_libraries(${TEST_TARGET} PRIVATE cuda-kat cudart ${CUDA_API_WRAPPERS_LIB} doctest  ${CUDA_LIBRARIES} test_utils)
	# I don't see why the following line should even be necessary. Depending on the libraries should be enough to get us their include dirs!
	target_include_directories(${TEST_TARGET} PRIVATE ${CUDA_API_WRAPPERS_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/src)

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
endforeach(TEST_TARGET)

if (BUILD_PRINTING_RELATED_TESTS)
	target_link_libraries(printing PRIVATE strf::strf-header-only)
endif()

if (KEEP_PTX)
	add_library(test_obj OBJECT ${tests})
	target_link_libraries(test_obj PRIVATE cuda-kat cudart ${CUDA_API_WRAPPERS_LIB} doctest)
	target_include_directories(test_obj PRIVATE ${CUDA_API_WRAPPERS_INCLUDE_DIR})
	target_compile_options(test_obj PRIVATE --expt-relaxed-constexpr)

	# Note: The constexpr_math test doesn't have any kernel code

	set_property(TARGET test_obj PROPERTY CUDA_PTX_COMPILATION ON)
	set_property(TARGET test_obj PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bin/tests)
endif()

# TODO: Something about code coverage perhaps?
