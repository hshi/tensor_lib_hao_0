set(time_exe tensortiming)

file(GLOB time_srcs "*.cpp")

add_executable(${time_exe} ${time_srcs})

target_link_libraries(${time_exe} tensorhao)

target_link_libraries(${time_exe} ${lib_all})

install(TARGETS ${time_exe}  DESTINATION timing)
