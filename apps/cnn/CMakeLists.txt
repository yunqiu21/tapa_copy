include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/apps.cmake)

add_executable(cnn)
target_sources(cnn PRIVATE cnn-host.cpp cnn.cpp)
target_link_libraries(cnn PRIVATE ${TAPA})
add_test(NAME cnn COMMAND cnn)

set(PLATFORM
  xilinx_u200_qdma_201920_1
)

find_package(SDx)
if(SDx_FOUND)
  add_tapa_target(
    cnn-hw-xo
    INPUT cnn.cpp
    TOP Cnn
    CONNECTIVITY ${CMAKE_CURRENT_SOURCE_DIR}/link_config.ini
    PLATFORM ${PLATFORM})

  add_xocc_hw_link_targets(
    ${CMAKE_CURRENT_BINARY_DIR}
    --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config.ini
    INPUT cnn-hw-xo
    HW_EMU_XCLBIN
    hw_emu_xclbin
    HW_XCLBIN
    hw_xclbin)

  add_custom_target(
    cnn-cosim
    COMMAND $<TARGET_FILE:cnn> 20
            --bitstream=$<TARGET_PROPERTY:${hw_emu_xclbin},FILE_NAME>
    DEPENDS cnn ${hw_emu_xclbin}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(
    cnn-hw
    COMMAND $<TARGET_FILE:cnn>
            --bitstream=$<TARGET_PROPERTY:${hw_xclbin},FILE_NAME>
    DEPENDS cnn ${hw_xclbin}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_test(NAME cnn-cosim COMMAND ${CMAKE_COMMAND} --build
                                     ${CMAKE_BINARY_DIR} --target cnn-cosim)
endif()
