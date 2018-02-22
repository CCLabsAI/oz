cmake_minimum_required (VERSION 3.1)
project (oz)

set (CMAKE_CXX_STANDARD 14)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wpedantic")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

set (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(PythonInterp)
find_package(PyTorch)

add_subdirectory("lib/pybind11" EXCLUDE_FROM_ALL)

if(PYTORCH_FOUND)
	pybind11_add_module(_ext "oz/csrc/oz.cpp")
	# set_target_properties(_ext PROPERTIES PREFIX "" DEBUG_POSTFIX "")
	target_include_directories(_ext PRIVATE "${PYTORCH_INCLUDE_DIR}" "lib/torch-stub/include")
endif()
