add_library(hello_cpu hello_cpu.cpp)

add_library(hello_gpu hello_gpu.cu)
target_compile_features(hello_gpu PUBLIC cxx_std_17)
set_target_properties( hello_gpu
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(main main.cu)
target_link_libraries(main hello_cpu hello_gpu)

add_executable(01-nbody_cpu 01-nbody_cpu.cu)
target_link_libraries(01-nbody_cpu fmt)