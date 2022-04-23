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

option(BUILD_PRINTF "Build with (non-header-only) printf-family functions support" OFF)
option(BUILD_TESTS "Build tests for the library" OFF)

# Ugly-but-necessary CUDA architectures detection

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})


###############
##  Targets  ##
###############

add_library(cuda-kat INTERFACE)
set(targets cuda-kat)
if (BUILD_PRINTF)
	add_library(kat-printf STATIC)
    set_target_properties(kat-printf PROPERTIES LINKER_LANGUAGE CUDA)

	target_sources(kat-printf PRIVATE src/kat/on_device/c_standard_library/printf.cu)
	#	set_property(TARGET kat-printf PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set_target_properties(
        kat-printf
        PROPERTIES
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
    )
	target_include_directories(kat-printf PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>")
	list(APPEND targets kat-printf)
endif()

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
	#REGEX "on_device/c_standard_library/printf\\.cuh" EXCLUDE
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
