# Copyright 2017 Everett Kropf.
#
# This file is part of ModifiedSchwarz.
#
# ModifiedSchwarz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModifiedSchwarz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ModifiedSchwarz.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Defaulting to CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS true)

set(CMAKE_CXX_FLAGS "-Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# For ubuntu 18.04 build
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_library(FMM_LIB hfmm2d)
if(NOT FMM_LIB)
    message(FATAL_ERROR "FMM library not found.")
else()
    message(STATUS "Found FMM library: ${FMM_LIB}")
endif()

find_library(GFORTRAN_LIB gfortran)
if(NOT GFORTRAN_LIB)
    message(FATAL_ERROR "gFortran lib not found.")
else()
    message(STATUS "Found gFortran: ${GFORTRAN_LIB}")
endif()

######################################################################
set(SCHWARZ_SOURCE
    src/BoundaryPoints.cpp
    src/CauchyInterpolant.cpp
    src/ComplexInterpolant.cpp
    src/ClosureInterpolant.cpp
    src/Problem.cpp
    src/RealInterpolant.cpp
    src/Solution.cpp
    src/Solver.cpp
    src/SpectralConstants.cpp
    src/SpectralData.cpp
    src/SpectralMethod.cpp
    src/UnitCircleDomain.cpp
    src/ZFMM_Wrapper.cpp
    )
add_library(schwarz_obj OBJECT ${SCHWARZ_SOURCE})
set_target_properties(${schwarz_obj} PROPERTIES POSITION_INDEPENDENT_CODE 1)

add_library(schwarz SHARED $<TARGET_OBJECTS:schwarz_obj>)
add_library(schwarz_static STATIC $<TARGET_OBJECTS:schwarz_obj>)
set_target_properties(schwarz_static PROPERTIES OUTPUT_NAME schwarz)
target_link_libraries(schwarz
    ${ARMADILLO_LIBRARIES}
    ${FMM_LIB}
    ${GFORTRAN_LIB}
    )

######################################################################
# Testing.

# Because MacPorts does not install the cmake file needed to use find_package().
if(NOT UNITTEST_FOUND)
    find_library(UNITTEST_LIBRARY UnitTest++)
    if(NOT UNITTEST_LIBRARY)
        message(FATAL_ERROR "UnitTest++ library not found.")
    else()
        message(STATUS "Found UnitTest++: " ${UNITTEST_LIBRARY})
    endif()

    get_filename_component(UNITTEST_LIBRARY_DIR ${UNITTEST_LIBRARY} DIRECTORY)
    find_path(UNITTEST_INCLUDE_DIR UnitTest++.h
        PATH_SUFFIXES unittest++ UnitTest++
        )
    if(NOT UNITTEST_INCLUDE_DIR)
        message(FATAL_ERROR "UnitTest++.h not found.")
    else()
        message(STATUS "Found UnitTest++.h: " "${UNITTEST_INCLUDE_DIR}/UnitTest++.h")
    endif()

    set(UNITTEST_FOUND ON CACHE INTERNAL BOOL)
endif()

######################################################################
if(UNITTEST_FOUND)
    set(TESTDIR "${CMAKE_CURRENT_SOURCE_DIR}/test")
    set(TESTSRC
        test/testRunner.cpp
        test/checkPolynomial.cpp
        test/checkUnitDomain.cpp
        test/checkBoundaryValues.cpp
        test/checkInterpolant.cpp
        test/checkFMM.cpp
        test/checkCauchyInterpolant.cpp
        test/checkClosureInterpolant.cpp
        test/checkSpectralConstants.cpp
        test/checkSpectralData.cpp
        test/checkSpectralMethod.cpp
        test/checkProblem.cpp
        test/checkSolution.cpp
        )

    ###
    add_executable(testRunner EXCLUDE_FROM_ALL ${TESTSRC})
    target_include_directories(testRunner
        PUBLIC ${CMAKE_SOURCE_DIR}/src
        PUBLIC ${UNITTEST_INCLUDE_DIR}
        )
    target_link_libraries(testRunner schwarz ${UNITTEST_LIBRARY})

    add_custom_target(check
        COMMAND ${CMAKE_BINARY_DIR}/testRunner
        DEPENDS testRunner
        )

endif(UNITTEST_FOUND)

###
add_executable(matWrite EXCLUDE_FROM_ALL test/matWrite.cpp)
target_include_directories(matWrite
    PUBLIC ${CMAKE_SOURCE_DIR}/src
    )
target_link_libraries(matWrite schwarz)

add_custom_target(write
    COMMAND ${CMAKE_BINARY_DIR}/matWrite
    DEPENDS matWrite
    )

######################################################################
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_BINARY_DIR}/Doxyfile @ONLY
        )
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API docs with Doxygen"
        )
endif()

######################################################################
# example
set(EXDIR "${CMAKE_SOURCE_DIR}/example")
set(EXSRC example/ex1.cpp)
add_executable(ex1 EXCLUDE_FROM_ALL ${EXSRC})
target_include_directories(ex1 PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ex1 schwarz)
add_custom_target(example
        COMMAND ${CMAKE_BINARY_DIR}/ex1
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ex1
        )
