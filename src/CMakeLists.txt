add_library(nbody_common nbody_common.cu)
target_compile_features(nbody_common PUBLIC cxx_std_17)
set_target_properties( nbody_common
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(nbody_gpu nbody_gpu.cu)
target_compile_features(nbody_gpu PUBLIC cxx_std_17)
set_target_properties( nbody_gpu
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(01-nbody_cpu 01-nbody_cpu.cu)
target_link_libraries(01-nbody_cpu 
  fmt
  nbody_common)