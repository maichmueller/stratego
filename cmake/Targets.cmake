cmake_minimum_required(VERSION 3.15)


set(STRATEGO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/stratego")
set(STRATEGO_CORE_FOLDER_SUFFIX "core/_core")
set(STRATEGO_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/stratego/${STRATEGO_CORE_FOLDER_SUFFIX}")
set(STRATEGO_CORE_SRC_DIR "${STRATEGO_CORE_DIR}/impl")
set(STRATEGO_CORE_INCLUDE_DIR ${STRATEGO_CORE_DIR}/include)
set(STRATEGO_INCLUDE_DIRS "${STRATEGO_CORE_INCLUDE_DIR};")

message(STATUS "STRATEGO project directory: ${STRATEGO_DIR}")
message(STATUS "STRATEGO CORE directory: ${STRATEGO_CORE_DIR}")
message(STATUS "STRATEGO CORE include directory: ${STRATEGO_CORE_INCLUDE_DIR}")
message(STATUS "STRATEGO CORE src directory: ${STRATEGO_CORE_SRC_DIR}")

add_library(core_lib STATIC)

set(CORE_LIBRARY_SOURCES
        logic.cpp
        )

list(TRANSFORM CORE_LIBRARY_SOURCES PREPEND "${STRATEGO_CORE_SRC_DIR}/")

target_sources(core_lib PRIVATE ${CORE_LIBRARY_SOURCES})

target_include_directories(core_lib
        PRIVATE
        ${STRATEGO_CORE_INCLUDE_DIR}
        ${CONAN_INCLUDE_DIRS_PYBIND11}
        )

set_target_properties(core_lib PROPERTIES
        CXX_STANDARD 17
        CXX_VISIBILITY_PRESET hidden
        )

target_compile_features(core_lib PRIVATE cxx_std_17)

target_link_libraries(core_lib PUBLIC ${PYTHON_LIBRARIES} project_options)

set(STRATEGO_CORE_BINDING_DIR ${STRATEGO_CORE_DIR}/binding)

set(PYTHON_MODULE_SOURCES
        module.cpp
        init_logic.cpp
)

list(TRANSFORM PYTHON_MODULE_SOURCES PREPEND "${STRATEGO_CORE_BINDING_DIR}/")

pybind11_add_module(_core_lib ${CORE_LIBRARY_SOURCES} ${PYTHON_MODULE_SOURCES})

set_target_properties(_core_lib PROPERTIES
        CXX_STANDARD 17
        CXX_VISIBILITY_PRESET hidden
        )
target_include_directories(_core_lib
        PRIVATE
        ${STRATEGO_CORE_INCLUDE_DIR}
        )


if (SKBUILD)
    message("Building with scikit-build. Disabling Test target.")
    # install locally so that sciki-build can correctly install it
    install(TARGETS _core_lib LIBRARY DESTINATION ./${STRATEGO_CORE_FOLDER_SUFFIX})

else ()
    if (ENABLE_TESTING)
        enable_testing()
        message(
                "Building Tests."
        )
        include(${CMAKE_CONFIG_FOLDER}/TargetsTest.cmake)
    endif ()

    # in order to use the latest build of the library for a development package install (pip install . -e),
    # we have to install it in the package folder, where it is used
    install(TARGETS _core_lib LIBRARY DESTINATION ${STRATEGO_CORE_DIR})
endif ()