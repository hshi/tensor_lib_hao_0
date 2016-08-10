#For diag_cpu
add_executable(diag_cpu diag_cpu.cpp)

target_link_libraries(diag_cpu tensorhao)

target_link_libraries(diag_cpu ${lib_all})

install(TARGETS diag_cpu  DESTINATION bin)

#For diag_magma
add_executable(diag_magma diag_magma.cpp)

target_link_libraries(diag_magma tensorhao)

target_link_libraries(diag_magma ${lib_all})

install(TARGETS diag_magma DESTINATION bin)
