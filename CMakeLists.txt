#
# Note:
#
# cuda-kat is a _header-only_ library. You can't build it, and you don't need
# to run CMake in order to use it or install it. Just add the `src/` directory
# to your include path (or copy its contents to some `include/` directory.
# cuda-kat only depends on having a C++11 compiler and the CUDA toolkit
# installed.
#
# This file is provided mostly in order to build the library unit tests.

cmake_minimum_required(VERSION 3.8.2)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

project(cuda-kat LANGUAGES CXX CUDA)
set(LIBRARY_NAME cuda-kat)

###############
##  Modules  ##
###############

# Standard CMake modules

# Custom modules

############################
##  Package dependencies  ##
############################

# cuda-kat can't use the standard library's string formatting and output stream code,
# because most of it is host-side only; and it doesn't make sense to bundle a modified
# half of the standard library just for that. Instead, we use a modification of strf, 
# from here: https://github.com/robhz786/strf
find_package(strf)

# TODO: Warn if strf is not found 


###############
##  OPTIONS  ##
###############

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
#message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(BUILD_TESTS FALSE CACHE BOOL "Build tests for the library")
set(KEEP_PTX FALSE CACHE BOOL "Keep kernel PTX files for build targets")


###############
##  Targets  ##
###############

add_library(cuda-kat INTERFACE)
target_include_directories(
    cuda-kat
    INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>"
    "$<INSTALL_INTERFACE:include>"
)


# TODO: Consider enabling the following command. It helps IDEs
# notice the library's header files even if they're not currently
# in use.
#
#target_sources(cuda-kat
#	src/kat/on_device/time.cuh
#	src/kat/on_device/shared_memory.cuh
#	 etc. etc.


#############
##  Tests  ##
#############

if(BUILD_TESTS)
    enable_testing()
    # set(TEST_RUNNER_PARAMS "--force-colors=true" CACHE STRING "Options to add to our test runners commands")
    add_subdirectory(tests)
endif()


####################
##  Installation  ##
####################

# WRITE ME!
