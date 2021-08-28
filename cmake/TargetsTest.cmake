
find_package("GTest")

set(STRATEGO_TEST_FOLDER "${CMAKE_SOURCE_DIR}/tests")

set(TEST_SOURCES
        tests.cpp
        )

list(TRANSFORM TEST_SOURCES PREPEND "${STRATEGO_TEST_FOLDER}/cpp/")

add_executable(tests ${TEST_SOURCES})

set_target_properties(tests PROPERTIES
        CXX_STANDARD 17
        CXX_VISIBILITY_PRESET hidden
        EXCLUDE_FROM_ALL True  # don't build tests when ALL is asked to be built. Only on demand.
        )

target_include_directories(tests PRIVATE
        ${STRATEGO_INCLUDE_DIRS}
        ${CONAN_INCLUDE_DIRS_GTEST}
        ${CONAN_INCLUDE_DIRS_PYBIND11})

target_link_libraries(tests PRIVATE
        # stratego libs
        core_lib
        # dependencies
        CONAN_PKG::gtest
        pybind11::embed
        ${Python_LIBRARIES})

