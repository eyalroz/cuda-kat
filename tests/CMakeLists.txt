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

add_executable(shared_memory shared_memory.cu)
target_link_libraries(shared_memory PRIVATE ${LIBRARY_NAME} cudart ${CUDA_API_WRAPPERS_LIB} doctest  ${CUDA_LIBRARIES})
target_include_directories(shared_memory PRIVATE ${CUDA_API_WRAPPERS_INCLUDE_DIR})
add_test(
    NAME ${PROJECT_NAME}.shared_memory
    COMMAND shared_memory ${TEST_RUNNER_PARAMS}
)

add_executable(constexpr_math constexpr_math.cu)
target_link_libraries(constexpr_math PRIVATE ${LIBRARY_NAME} cudart ${CUDA_API_WRAPPERS_LIB} doctest  ${CUDA_LIBRARIES})
target_include_directories(constexpr_math PRIVATE ${CUDA_API_WRAPPERS_INCLUDE_DIR})
add_test(
	NAME ${PROJECT_NAME}.constexpr_math
	COMMAND constexpr_math ${TEST_RUNNER_PARAMS}
)


set_target_properties(
#	failtest
	shared_memory
	constexpr_math
	PROPERTIES 
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES 
    CXX_EXTENSIONS NO
)

if (KEEP_PTX)
	add_library(shared_memory_obj OBJECT shared_memory.cu)
	target_link_libraries(shared_memory_obj ${LIBRARY_NAME} cudart ${CUDA_API_WRAPPERS_LIB} doctest)
	target_include_directories(shared_memory_obj PRIVATE ${CUDA_API_WRAPPERS_INCLUDE_DIR})

	# Note: The constexpr_math test doesn't have any kernel code

	set_property(TARGET shared_memory_obj PROPERTY CUDA_PTX_COMPILATION ON)
	set_property(TARGET shared_memory_obj PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bin/tests)
endif()

# TODO: Something about code coverage perhaps?
