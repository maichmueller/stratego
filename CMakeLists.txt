cmake_minimum_required(VERSION 3.15)

project(STRATEGO VERSION 0.1 LANGUAGES CXX)
include(cmake/StandardProjectSettings.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if (APPLE)
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

message("C++ Compiler: ${CMAKE_CXX_COMPILER}")
message("C++ Compiler Info: ${CMAKE_CXX_COMPILER_ID}, version ${CMAKE_CXX_COMPILER_VERSION}")
message("CURRENT CMAKE BINARY DIR: ${CMAKE_CURRENT_BINARY_DIR}")
if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)
    if (ENABLE_BUILD_WITH_TIME_TRACE)
        add_compile_definitions(project_options INTERFACE -ftime-trace)
    endif ()
endif ()

set(CMAKE_CONFIG_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set this path if the conan install is not in the standard directory
# This doesn't work due to a conan internal error when conan is not found in the standard directory (/usr/bin on linux)
# It doesn't work in general when CONAN_COMMAND is set it seems.
#execute_process(COMMAND which conan OUTPUT_VARIABLE CONAN_PATH)


# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)
# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# enable cache system
include(${CMAKE_CONFIG_FOLDER}/Cache.cmake)

# standard compiler warnings
include(${CMAKE_CONFIG_FOLDER}/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options if supported by compiler
include(${CMAKE_CONFIG_FOLDER}/Sanitizers.cmake)
enable_sanitizers(project_options)

# enable doxygen
include(${CMAKE_CONFIG_FOLDER}/Doxygen.cmake)
option(ENABLE_DOXYGEN "Enable doxygen doc builds of source")
enable_doxygen(ENABLE_DOXYGEN)

# allow for static analysis options
include(${CMAKE_CONFIG_FOLDER}/StaticAnalyzers.cmake)

option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)
option(ENABLE_TESTING "Enable Test Builds" ON)
option(ENABLE_FUZZING "Enable Fuzzing Builds" OFF)

# Very basic PCH example
option(ENABLE_PCH "Enable Precompiled Headers" ON)
if (ENABLE_PCH)
    # This sets a global PCH parameter, each project will build its own PCH, which
    # is a good idea if any #define's change
    #
    # consider breaking this out per project as necessary
    target_precompile_headers(project_options
            INTERFACE <vector> <string> <map> <utility> <memory> <array> <list> <stack>)
endif ()

# Set up some extra Conan dependencies based on our needs
# before loading Conan
set(CONAN_EXTRA_REQUIRES "")
set(CONAN_EXTRA_OPTIONS "")

include(${CMAKE_CONFIG_FOLDER}/Conan.cmake)
run_conan()
include(${PROJECT_BINARY_DIR}/conanbuildinfo.cmake)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

set(STRATEGO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/stratego")
set(STRATEGO_SRC_DIR "${STRATEGO_DIR}/core/impl")
set(STRATEGO_INCLUDE_DIR ${STRATEGO_DIR}/core/include)

message(STATUS "STRATEGO project directory: ${STRATEGO_DIR}")
message(STATUS "STRATEGO include directory: ${STRATEGO_INCLUDE_DIR}")
message(STATUS "STRATEGO src directory: ${STRATEGO_SRC_DIR}")

add_library(stratego_cpp_lib STATIC)

# this could be used:
#find_package(Python COMPONENTS Interpreter Development)
# but better to just let pybind11 do it:
set(PYBIND11_FINDPYTHON TRUE)
find_package(pybind11 REQUIRED)

add_subdirectory(src)

if (SKBUILD)
    message("Building with scikit-build. Disabling Test target.")
    # install locally so that sciki-build can correctly install it
    install(TARGETS _core DESTINATION ./core)

else ()
    if (ENABLE_TESTING)
        enable_testing()
        message(
                "Building Tests."
        )
        add_subdirectory(test)
    endif ()

    # in order to use the latest build of the library for a development package install (pip install . -e),
    # we have to install it in the package folder, where it is used
    install(TARGETS _core LIBRARY DESTINATION ${STRATEGO_DIR}/core)
endif ()

