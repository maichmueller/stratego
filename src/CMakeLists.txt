cmake_minimum_required(VERSION 3.15)

set(LIBRARY_SOURCES
        ${STRATEGO_SRC_DIR}/logic.cpp
        )

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

target_sources(stratego_cpp_lib PRIVATE ${LIBRARY_SOURCES})

target_include_directories(stratego_cpp_lib
        PRIVATE
        ${STRATEGO_INCLUDE_DIR}
        ${CONAN_INCLUDE_DIRS_PYBIND11}
        ${Python_INCLUDE_DIRS}
        )

set_target_properties(stratego_cpp_lib PROPERTIES
        CXX_STANDARD 17
        CXX_VISIBILITY_PRESET hidden
        )

target_compile_features(stratego_cpp_lib PRIVATE cxx_std_17)

target_link_libraries(stratego_cpp_lib PUBLIC ${PYTHON_LIBRARIES} project_options)

set(STRATEGO_BINDING_DIR ${STRATEGO_DIR}/core/binding)

set(PYTHON_MODULE_SOURCES
        ${STRATEGO_BINDING_DIR}/module.cpp
        ${STRATEGO_BINDING_DIR}/init_logic.cpp
)

pybind11_add_module(_core ${LIBRARY_SOURCES} ${PYTHON_MODULE_SOURCES})

set_target_properties(_core PROPERTIES
        CXX_STANDARD 17
        CXX_VISIBILITY_PRESET hidden
        )
target_include_directories(_core
        PRIVATE
        ${STRATEGO_INCLUDE_DIR}
        )
