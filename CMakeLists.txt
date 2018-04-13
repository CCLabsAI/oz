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
        "oz/csrc/tests/test_oos.cpp"
        "oz/csrc/oos.cpp"
        "oz/csrc/tests/test_leduk.cpp"
        "oz/csrc/games/leduk.cpp"
        )

add_executable("solve_leduk"
        "oz/csrc/solve_leduk.cpp"
        "oz/csrc/games/leduk.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oos.cpp"
        )

pybind11_add_module(_ext
        "oz/csrc/oz.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oos.cpp"
        "oz/csrc/batch.cpp"
        "oz/csrc/encoder.cpp"
        "oz/csrc/py_sigma.cpp"
        "oz/csrc/games/flipguess.cpp"
        "oz/csrc/games/kuhn.cpp"
        "oz/csrc/games/leduk.cpp"
        )
#pybind11_add_module(_ext "oz/csrc/oz.cpp")

target_include_directories(_ext SYSTEM PRIVATE "${PYTORCH_INCLUDE_DIR}" "lib/torch-stub/include")
