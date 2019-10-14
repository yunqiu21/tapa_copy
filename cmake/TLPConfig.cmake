function(add_tlp_target target_name)
  # Generate Xilinx object file from TLP C++.
  #
  # The added target will have the following properties:
  #
  # * FILE_NAME
  # * TARGET
  # * KERNEL
  # * PLATFORM
  #
  # Positional Arguments:
  #
  # * target_name: Name of the added cmake target.
  #
  # Required Named Arguments:
  #
  # * INPUT: Input filename.
  # * TOP: Name of the top-level task, which will be used the kernel name.
  # * PLATFORM: SDAccel platform name or path.
  #
  # Optional Named Arguments:
  #
  # * OUTPUT: Optional, output filename, default to ${TOP}.${PLATFORM}.hw.xo
  # * DRAM_MAPPING: A list of mappings from variable name to DDR banks (e.g.
  #   gmem0:DDR[0]).
  # * TLPC: Optional, path to the tlpc executable.
  # * TLPCC: Optional, path to the tlpcc executable.
  cmake_parse_arguments(TLP
                        ""
                        "OUTPUT;INPUT;TOP;PLATFORM;TLPC;TLPCC"
                        "DRAM_MAPPING"
                        ${ARGN})
  if(NOT TLP_PLATFORM)
    message(FATAL_ERROR "PLATFORM not specified")
  endif()
  if(NOT TLP_TOP)
    message(FATAL_ERROR "TOP not specified")
  endif()

  if(NOT TLP_OUTPUT)
    set(TLP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TLP_TOP}.${TLP_PLATFORM}.hw.xo)
  endif()
  get_filename_component(TLP_INPUT ${TLP_INPUT} ABSOLUTE)
  get_filename_component(TLP_OUTPUT ${TLP_OUTPUT} ABSOLUTE)

  if(TLP_TLPC)
    set(TLPC ${TLP_TLPC})
  endif()
  if(NOT TLPC)
    find_program(TLPC tlpc)
  endif()
  if(NOT TLPC)
    message(FATAL_ERROR "cannot find tlpc")
  endif()

  if(TLP_TLPCC)
    set(TLPCC ${TLP_TLPCC})
  endif()
  if(NOT TLPCC)
    find_program(TLPCC tlpcc)
  endif()
  if(NOT TLPCC)
    message(FATAL_ERROR "cannot find tlpcc")
  endif()

  set(tlpcc_cmd
      "${TLPC}"
      "${TLP_INPUT}"
      "--top=${TLP_TOP}"
      "--tlpcc=${TLPCC}"
      "--platform=${TLP_PLATFORM}"
      "--output=${TLP_OUTPUT}")

  add_custom_command(OUTPUT ${TLP_OUTPUT}
                     COMMAND ${tlpc_cmd}
                     DEPENDS ${TLP_INPUT}
                     VERBATIM)

  add_custom_target(${target_name} DEPENDS ${TLP_OUTPUT})
  set_target_properties(${target_name}
                        PROPERTIES FILE_NAME
                                   ${TLP_OUTPUT}
                                   TARGET
                                   hw
                                   KERNEL
                                   ${TLP_TOP}
                                   PLATFORM
                                   ${TLP_PLATFORM}
                                   DRAM_MAPPING
                                   "${TLP_DRAM_MAPPING}")
endfunction()
