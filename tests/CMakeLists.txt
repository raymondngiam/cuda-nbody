set(TEST_BINARY ${PROJECT_NAME}_test)
add_executable(${TEST_BINARY} test_main.cu)  
target_link_libraries(${TEST_BINARY} 
    nbody_common
    nbody_gpu
    fmt
    ${Boost_LIBRARIES}
    gtest 
    gtest_main)

add_test(NAME ${TEST_BINARY}
         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_BINARY})