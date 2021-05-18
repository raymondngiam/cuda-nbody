add_library(test_utils test_utils.cu)
target_link_libraries(test_utils nbody_common fmt ${Boost_LIBRARIES})

set(TEST_BINARY ${PROJECT_NAME}_test)
add_executable(${TEST_BINARY} 
    Test01_RefactorBodyForce.cu
    Test02_KernelIntegration.cu
    Test03_Prefetch.cu)  
target_link_libraries(${TEST_BINARY} 
    nbody_common
    nbody_gpu
    test_utils
    fmt    
    gtest 
    gtest_main)

add_test(NAME ${TEST_BINARY}
         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_BINARY})