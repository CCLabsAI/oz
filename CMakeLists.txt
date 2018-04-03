cmake_minimum_required (VERSION 3.1)
project (oz)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options("-Wall")
add_compile_options("$<$<CONFIG:Release>:-march=native>")

find_package(PythonInterp 3.4 REQUIRED)
find_package(PyTorch 0.3.0 REQUIRED)

add_subdirectory("lib/pybind11" EXCLUDE_FROM_ALL)
include_directories("lib/catch/include")

include_directories("oz/csrc")

add_executable("tests"
        "oz/csrc/tests/tests.cpp"
        "oz/csrc/tests/test_flipguess.cpp"
        "oz/csrc/games/flipguess.cpp"
        "oz/csrc/games/kuhn.cpp"
        "oz/csrc/tests/test_best_response.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oss.cpp"
        )

pybind11_add_module(_ext
        "oz/csrc/oz.cpp"
        "oz/csrc/games/flipguess.cpp"
        "oz/csrc/games/kuhn.cpp"
        )
#pybind11_add_module(_ext "oz/csrc/oz.cpp")

target_include_directories(_ext SYSTEM PRIVATE "${PYTORCH_INCLUDE_DIR}" "lib/torch-stub/include")
