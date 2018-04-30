unset(PYTORCH_VERSION)
unset(PYTORCH_INCLUDE_DIR)
unset(__result)
unset(__output)
unset(__ver_check)

if(PYTHONINTERP_FOUND)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import os, torch; print(';'.join([torch.__version__, os.path.abspath(os.path.join(os.path.dirname(torch.__file__), 'lib', 'include'))]))"
    RESULT_VARIABLE __result
    OUTPUT_VARIABLE __output
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(__result MATCHES 0)
    list(GET __output 0 PYTORCH_VERSION)
    list(GET __output 1 PYTORCH_INCLUDE_DIR)

    string(REGEX MATCH "^([0-9])+\\.([0-9])+\\.([0-9])+" __ver_check "${PYTORCH_VERSION}")
    if(NOT "${__ver_check}" STREQUAL "")
      set(PYTORCH_VERSION_MAJOR ${CMAKE_MATCH_1})
      set(PYTORCH_VERSION_MINOR ${CMAKE_MATCH_2})
      set(PYTORCH_VERSION_PATCH ${CMAKE_MATCH_3})
      math(EXPR PYTORCH_VERSION_DECIMAL
        "(${PYTORCH_VERSION_MAJOR} * 10000) + (${PYTORCH_VERSION_MINOR} * 100) + ${PYTORCH_VERSION_PATCH}")
    else()
       unset(PYTORCH_VERSION)
       unset(PYTORCH_INCLUDE_DIR)
       message(STATUS "Requested PyTorch version and include path, but got instead:\n${__output}\n")
    endif()
  endif()


else()
  message(STATUS "Unable to find PyTorch, Python interpreter is unavailable")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PyTorch
  REQUIRED_VARS PYTORCH_VERSION PYTORCH_INCLUDE_DIR
  VERSION_VAR PYTORCH_VERSION)


unset(__result)
unset(__output)
unset(__ver_check)
