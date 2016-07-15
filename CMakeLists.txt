cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(tensorhao)

#For find module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "~/lib/Modules")

#For definitions
add_definitions(${COMPILER_EXTRA_DEF})

#For c++ flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(inc)

#For library
set(lib_all "")

   # For lapackblas library
   find_package(mkl)
   find_package(acml)
   find_package(LAPACK)
   if(MKL_FOUND)
     message("-- Use MKL library." )
     add_definitions(-DUSE_MKL)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${mkl_flags}")
     include_directories(${mkl_include_dirs})
     set(lapackblas_libraries "${mkl_libraries}")
   elseif(ACML_FOUND)
     message("-- Use ACML library." )
     add_definitions(-DUSE_ACML)
     include_directories(${acml_include_dirs})
     set(lapackblas_libraries "${acml_libraries}")
   elseif(LAPACK_FOUND AND BLAS_FOUND)
     message("-- Use system lapack and blas library." )
     #Usually, Fortran has underscore
     #No need to turn on FORTRAN_NO_UNDERSCORE
     if(FORTRAN_NO_UNDERSCORE)
       add_definitions(-DFORTRAN_NO_UNDERSCORE)
     endif()
     add_definitions(-DUSE_BLAS_LAPACK)
     set(lapackblas_libraries ${LAPACK_LIBRARIES})
   else()
     message( FATAL_ERROR "Find no library for blas and lapack!" )
   endif()
   set(lib_all ${lapackblas_libraries} ${lib_all})
   
   #For magma library
   if(USE_MAGMA)
     add_definitions(-DUSE_MAGMA)
     # Use standard FindCUD
     find_package(CUDA REQUIRED)
     include_directories(${CUDA_INCLUDE_DIRS})
     find_package(magma REQUIRED)
     include_directories(${magma_include_dirs})
     set(magma_libraries "${magma_libraries}" "${CUDA_LIBRARIES}" "${CUDA_CUBLAS_LIBRARIES}")
   else()
     MESSAGE("-- MAGMA support was disabled.")
   endif()
   set(lib_all ${magma_libraries} ${lib_all})

add_subdirectory(inc)

add_subdirectory(src)

add_subdirectory(test)

add_subdirectory(timing)
