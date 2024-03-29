cmake_minimum_required(VERSION 3.15)

project(STRATEGO LANGUAGES CXX)

set(CMAKE_CONFIG_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(${CMAKE_CONFIG_FOLDER}/StandardProjectSettings.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
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


option(ENABLE_TESTING "Enable Building Tests" ON)

# Set up some extra Conan dependencies based on our needs
# before loading Conan
set(CONAN_EXTRA_REQUIRES "")
set(CONAN_EXTRA_OPTIONS "")

include(${CMAKE_CONFIG_FOLDER}/Conan.cmake)
run_conan()
include(${PROJECT_BINARY_DIR}/conanbuildinfo.cmake)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

# this could be used:
# find_package(Python COMPONENTS Interpreter Development)
# but better to just let pybind11 do it:
set(PYBIND11_FINDPYTHON TRUE)

if (SKBUILD OR DEFINED pybind11_path)
    find_package(pybind11 CONFIG PATHS ${pybind11_path} REQUIRED)
else ()
    find_package(pybind11 REQUIRED)
endif ()

include(${CMAKE_CONFIG_FOLDER}/Targets.cmake)
