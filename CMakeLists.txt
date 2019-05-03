cmake_minimum_required(VERSION 3.8.2)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

project(
	cuda-kat
	LANGUAGES CXX CUDA)
set(LIBRARY_NAME ${PROJECT_NAME})

###############
##  Modules  ##
###############

# Standard CMake modules

# Custom modules

###############
##  OPTIONS  ##
###############

#cmake_dependent_option(BUILD_CUDA_KAT_TESTS
#    "Enable Boilerplate project tests targets" ON # By default we want tests if CTest is enabled
#    "BUILD_TESTING" OFF # Stay coherent with CTest variables
#)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(BUILD_TESTING TRUE CACHE BOOL "Build tests for the library")
set(KEEP_PTX TRUE CACHE BOOL "Keep kernel PTX files for build targets")

#############
##  Tests  ##
#############

if(BUILD_TESTING)
	enable_testing()
    # Let the user add options to the test runner if needed
    # set(TEST_RUNNER_PARAMS "--force-colors=true" CACHE STRING "Options to add to our test runners commands")

    # In a real project you most likely want to exclude test folders
    # list(APPEND CUSTOM_COVERAGE_EXCLUDE "/test/")

    add_subdirectory(tests)

    # You can setup some custom variables and add them to the CTestCustom.cmake.in template to have custom ctest settings
    # For example, you can exclude some directories from the coverage reports such as third-parties and tests
    #configure_file(
    #    ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestCustom.cmake.in
    #    ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake
    #    @ONLY
    #)
endif()


###############
##  Targets  ##
###############

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE src/)

# TODO: Consider enabling the following command. It helps IDEs
# notice the library's header files even if they're not currently
# in use.
#
#target_sources(${LIBRARY_NAME}
#	src/kat/on_device/time.cuh
#	src/kat/on_device/shared_memory.cuh
#	 etc. etc.

