cmake_minimum_required(VERSION 3.9)
project(oz)

add_subdirectory("lib/pybind11" EXCLUDE_FROM_ALL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckIPOSupported)

add_compile_options("-Wall")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

add_compile_options("$<$<CONFIG:Release>:-march=native>")
add_compile_options("$<$<CONFIG:RelWithDebInfo>:-march=native>")

find_package(PythonInterp 3.4 REQUIRED)
find_package(PyTorch 0.4.0 REQUIRED)

include_directories(SYSTEM "lib/boost/include")
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

        "oz/csrc/tests/test_goofspiel2.cpp"
        "oz/csrc/games/goofspiel2.cpp"

        "oz/csrc/tests/test_target.cpp"
        "oz/csrc/target/leduk_target.cpp"
        "oz/csrc/target/goofspiel2_target.cpp"

        "oz/csrc/oos_rec.cpp"
        )

target_include_directories(tests SYSTEM PRIVATE "lib/catch/include")


add_executable("solve_leduk"
        "oz/csrc/solve_leduk.cpp"
        "oz/csrc/games/leduk.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oos.cpp"
        )

add_executable("solve_goofspiel2"
        "oz/csrc/solve_goofspiel2.cpp"
        "oz/csrc/games/goofspiel2.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oos.cpp"
        )

pybind11_add_module(_ext
        "oz/csrc/oz.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oos.cpp"
        "oz/csrc/batch.cpp"
        "oz/csrc/py_sigma.cpp"
        "oz/csrc/games/flipguess.cpp"
        "oz/csrc/games/kuhn.cpp"

        "oz/csrc/games/leduk.cpp"
        "oz/csrc/encoder/leduk_encoder.cpp"
        "oz/csrc/target/leduk_target.cpp"

        "oz/csrc/games/goofspiel2.cpp"
        "oz/csrc/encoder/goofspiel2_encoder.cpp"
        "oz/csrc/target/goofspiel2_target.cpp"
        )

target_include_directories(_ext SYSTEM PRIVATE "${PYTORCH_INCLUDE_DIR}")

check_ipo_supported(RESULT result)
if(result)
  set_property(TARGET solve_leduk PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_property(TARGET solve_goofspiel2 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
