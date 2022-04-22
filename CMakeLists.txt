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

project(cuda-kat
	DESCRIPTION "CUDA kernel author's tools"
	VERSION 0.2.0
	HOMEPAGE_URL "https://github.com/eyalroz/cuda-kat"
	LANGUAGES CXX CUDA)

###############
##  Modules  ##
###############

# Standard CMake modules

include(CMakeDependentOption)

# Custom modules

############################
##  Package dependencies  ##
############################

###############
##  OPTIONS  ##
###############

#message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


option(BUILD_TESTS "Build tests for the library" OFF)

###############
##  Targets  ##
###############

add_library(cuda-kat INTERFACE)
set(targets cuda-kat)

foreach(tgt ${targets})
	target_include_directories(
		${tgt}
		INTERFACE
		"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>"
		"$<INSTALL_INTERFACE:include>"
	)
endforeach()



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

include(GNUInstallDirs)

install(
		TARGETS ${targets}
		EXPORT cuda-kat_export
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
		ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
		LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
		INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
	DIRECTORY src/kat
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	FILES_MATCHING REGEX "\\.(h|hpp|cuh)$"
)

install(
	EXPORT cuda-kat_export
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuda-kat"
	NAMESPACE "cuda-kat::"
	FILE cuda-kat-config.cmake
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
	"cuda-kat-config-version.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMinorVersion
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/cuda-kat-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cuda-kat"
)
