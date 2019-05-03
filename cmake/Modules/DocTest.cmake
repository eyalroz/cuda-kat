add_library(doctest INTERFACE)
set(DOCTEST_DIR "${PROJECT_SOURCE_DIR}/external/doctest/")
target_sources(doctest INTERFACE ${DOCTEST_DIR}/doctest.h) # Is this needed?
target_include_directories(doctest INTERFACE ${DOCTEST_DIR})

