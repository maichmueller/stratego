cmake_minimum_required(VERSION 3.15)

#if(NOT TORCH_FOUND)
#	if(UNIX)
#		set(CMAKE_PREFIX_PATH ../deps/libtorch)
#		if(USE_CUDA)
#			set(CUDNN_LIBRARY /usr/local/cuda/lib64/libcudnn.so)
#			set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
#		ENDIF()
#	ENDIF(UNIX)
#	IF(WIN32)
#		IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
#			set(CMAKE_PREFIX_PATH deps/libtorch_debug)
#		ELSE()
#			set(CMAKE_PREFIX_PATH deps/libtorch)
#		ENDIF()
#	ENDIF(WIN32)
#
#	find_package(Torch REQUIRED)
#	# PARENT_SCOPE is necessary to set the torch_found variable of the main cmakelists
#	# and not a local only version of it
##	set(Torch_FOUND TRUE PARENT_SCOPE)
#ENDIF()


set(STRATEGO_SOURCES
		${STRATEGO_SRC_DIR}/board/PieceStratego.h
		${STRATEGO_SRC_DIR}/board/BoardStratego.cpp
		${STRATEGO_SRC_DIR}/game/GameStratego.cpp
		${STRATEGO_SRC_DIR}/game/StateStratego.cpp
		${STRATEGO_SRC_DIR}/nn/representation/RepresenterStratego.cpp
		${STRATEGO_SRC_DIR}/logic/LogicStratego.cpp
		${STRATEGO_SRC_DIR}/main.cpp
		)

add_executable(stratego ${STRATEGO_SOURCES})

target_include_directories(stratego
		PRIVATE
		${AZE_INCLUDE_DIR}
		${TORCH_INCLUDE_DIRS}
		${STRATEGO_INCLUDE_DIR}
		${TQDM_INCLUDE_DIR}
		)

set_target_properties(stratego PROPERTIES
		CXX_STANDARD 17
		)

#set(SANITIZE_OPTIONS -fsanitize=address -fsanitize=undefined)
target_compile_options(stratego PRIVATE -Wall ${SANITIZE_OPTIONS})
target_link_options(stratego PRIVATE ${SANITIZE_OPTIONS} --verbose)

target_link_libraries(stratego aze)
target_link_libraries(stratego "${TORCH_LIBRARIES}")
target_link_libraries(stratego "${CUDNN_LIBRARY}")
# stdc++fs is needed for static linking against the std::filesystem library (not needed for GCC >=9)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9")
	target_link_libraries(stratego stdc++fs)
endif()