cmake_minimum_required(VERSION 3.5)
project(oz LANGUAGES C CXX)

add_subdirectory("lib/pybind11" EXCLUDE_FROM_ALL)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#include(CheckIPOSupported)

add_compile_options("-Wall")

# NB hide all symbols, we only want to export the _PyInit entry point
add_compile_options("-fvisibility=hidden")

# NB fast changes results, due to floating point optimizations
# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
# TODO move critical correctness checks out of asserts?
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

set(CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})

add_compile_options("$<$<CONFIG:Release>:-march=native>")
add_compile_options("$<$<CONFIG:RelWithDebInfo>:-march=native>")

find_package(PythonInterp 3.4 REQUIRED)
find_package(PyTorch 0.4.0 REQUIRED)

include_directories(SYSTEM "lib/gsl-lite/include")
include_directories(SYSTEM "lib/boost/include")
include_directories(SYSTEM "lib/ACE_Eval/include")

include_directories("oz/csrc")

add_executable(tests
        "oz/csrc/tests/tests.cpp"

        "oz/csrc/tests/test_flipguess.cpp"
        "oz/csrc/games/flipguess.cpp"

        "oz/csrc/games/kuhn.cpp"

        "oz/csrc/tests/test_best_response.cpp"
        "oz/csrc/best_response.cpp"

        "oz/csrc/tests/test_oos.cpp"
        "oz/csrc/oos.cpp"
        "oz/csrc/history.cpp"
        "oz/csrc/sigma.cpp"
        "oz/csrc/node.cpp"
        "oz/csrc/tree.cpp"

        "oz/csrc/tests/test_leduc.cpp"
        "oz/csrc/games/leduc.cpp"

        "oz/csrc/tests/test_goofspiel2.cpp"
        "oz/csrc/games/goofspiel2.cpp"

        "oz/csrc/tests/test_holdem.cpp"
        "oz/csrc/games/holdem.cpp"
        "oz/csrc/target/holdem_target.cpp"

        "oz/csrc/tests/test_target.cpp"
        "oz/csrc/target/leduc_target.cpp"
        "oz/csrc/target/goofspiel2_target.cpp"

        "oz/csrc/tests/test_mcts.cpp"
        "oz/csrc/mcts.cpp"

        "oz/csrc/oos_rec.cpp"
        )


target_include_directories(tests SYSTEM PRIVATE "lib/catch/include")

# NB alloc_lib.c includes other c files these should not be included twice
file(GLOB boost_container_SRC
        "lib/boost/libs/container/src/alloc_lib.c"
        "lib/boost/libs/container/src/*.cpp"
        )

add_library(boost_container STATIC ${boost_container_SRC})
set_property(TARGET boost_container PROPERTY POSITION_INDEPENDENT_CODE ON)

# NB prevents the boost internal allocator DLMalloc from using sbrk
target_compile_options(boost_container PRIVATE "-DHAVE_MORECORE=0")

add_library(ACE_Eval STATIC "lib/ACE_Eval/ace_eval.cpp")
set_property(TARGET ACE_Eval PROPERTY POSITION_INDEPENDENT_CODE ON)
set_source_files_properties("lib/ACE_Eval/ace_eval.cpp" PROPERTIES COMPILE_FLAGS
    "-Wno-parentheses -Wno-shift-op-parentheses")

add_executable(solve_leduc
        "oz/csrc/solve_leduc.cpp"
        "oz/csrc/games/leduc.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oos.cpp"
        "oz/csrc/history.cpp"
        "oz/csrc/sigma.cpp"
        "oz/csrc/node.cpp"
        "oz/csrc/tree.cpp"
        )

add_executable(solve_leduc_mcts
        "oz/csrc/solve_leduc_mcts.cpp"
        "oz/csrc/games/leduc.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/mcts.cpp"
        "oz/csrc/history.cpp"
        "oz/csrc/sigma.cpp"
        )

add_executable(solve_goofspiel2
        "oz/csrc/solve_goofspiel2.cpp"
        "oz/csrc/games/goofspiel2.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oos.cpp"
        "oz/csrc/history.cpp"
        "oz/csrc/sigma.cpp"
        "oz/csrc/node.cpp"
        "oz/csrc/tree.cpp"
        )

pybind11_add_module(_ext
        "oz/csrc/oz.cpp"
        "oz/csrc/best_response.cpp"
        "oz/csrc/oos.cpp"
        "oz/csrc/history.cpp"
        "oz/csrc/sigma.cpp"
        "oz/csrc/node.cpp"
        "oz/csrc/tree.cpp"
        "oz/csrc/batch.cpp"
        "oz/csrc/mcts.cpp"
        "oz/csrc/mcts_batch.cpp"
        "oz/csrc/py_sigma.cpp"
        "oz/csrc/py_sigma_batch.cpp"
        "oz/csrc/games/flipguess.cpp"
        "oz/csrc/games/kuhn.cpp"

        "oz/csrc/games/leduc.cpp"
        "oz/csrc/encoder/leduc_encoder.cpp"
        "oz/csrc/target/leduc_target.cpp"

        "oz/csrc/games/goofspiel2.cpp"
        "oz/csrc/encoder/goofspiel2_encoder.cpp"
        "oz/csrc/target/goofspiel2_target.cpp"

        "oz/csrc/games/holdem.cpp"
        "oz/csrc/encoder/holdem_encoder.cpp"
        "oz/csrc/target/holdem_target.cpp"
        )

target_include_directories(_ext SYSTEM PRIVATE "${PYTORCH_INCLUDE_DIR}")

#check_ipo_supported(RESULT result)
#if(result)
  #set_property(TARGET boost_container PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  #set_property(TARGET solve_leduc PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  #set_property(TARGET solve_goofspiel2 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
#endif()

target_link_libraries(tests PRIVATE boost_container ACE_Eval)
target_link_libraries(solve_leduc PRIVATE boost_container)
target_link_libraries(solve_leduc_mcts PRIVATE boost_container)
target_link_libraries(solve_goofspiel2 PRIVATE boost_container)
target_link_libraries(_ext PRIVATE boost_container ACE_Eval)
