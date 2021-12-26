set(TIMVX_COMMIT_HASH "19c0bd820a83a2bede6e1999f833f915f8801bae")
set(OCV_TIMVX_DIR "${OpenCV_BINARY_DIR}/3rdparty/libtim-vx")
set(OCV_TIMVX_SOURCE_PATH "${OCV_TIMVX_DIR}/TIM-VX-${TIMVX_COMMIT_HASH}")

set(OCV_VIVANTE_DIR "${OCV_TIMVX_DIR}/VIVANTE_SDK")

if(EXISTS "${OCV_TIMVX_SOURCE_PATH}")
    message(STATUS "TIM-VX: Use cache at ${OCV_TIMVX_SOURCE_PATH}")

    set(TIMVX_FOUND ON)
    set(BUILD_TIMVX ON)
else()
    set(OCV_TIMVX_FILENAME "${TIMVX_COMMIT_HASH}.zip")
    set(OCV_TIMVX_URL "https://github.com/fengyuentau/TIM-VX/archive/")
    set(timvx_zip_md5sum 78435539278454aa4d85b886f06bc454)

    ocv_download(FILENAME ${OCV_TIMVX_FILENAME}
                 HASH ${timvx_zip_md5sum}
                 URL "${OCV_TIMVX_URL}"
                 DESTINATION_DIR "${OCV_TIMVX_DIR}"
                 ID TIMVX
                 STATUS res
                 UNPACK RELATIVE_URL)

    if(NOT res)
        set(TIMVX_FOUND OFF)
        message(STATUS "TIM-VX: Failed to download source code from github. Turning off TIMVX_FOUND")
    else()
        set(TIMVX_FOUND ON)
        set(BUILD_TIMVX ON)
        message(STATUS "TIM-VX: Source code downloaded at ${OCV_TIMVX_SOURCE_PATH}.")
    endif()
endif()

if(BUILD_TIMVX)
    set(HAVE_TIMVX 1)

    ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter -Wstrict-prototypes -Wundef -Wsign-compare -Wmissing-prototypes -Wmissing-declarations -Wstrict-aliasing -Wunused-but-set-variable -Wmaybe-uninitialized -Wshadow -Wsuggest-override)
    ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter -Wstrict-prototypes -Wundef -Wsign-compare -Wunused-but-set-variable -Wshadow -Wsuggest-override -Wmissing-declarations)

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
        set(vivante_sdk_filename "aarch64;6.4.8.tgz")
        set(vivante_sdk_A311D_md5sum da530e28f73fd8b143330b6d1b97a1d8)
        set(vivante_sdk_S905D3_md5sum f89ae2b52e53c4a8d5d3fb1e8b3bbcf9)

        #set(VIVANTE_SDK_FOR "A311D" CACHE STRING "Either A311D or S905D3")
        execute_process(
            COMMAND
              cat /proc/cpuinfo
            RESULT_VARIABLE
              timvx_result_var
            OUTPUT_VARIABLE
              timvx_output_var
        )
        string(FIND "${timvx_output_var}" "Khadas VIM3" _found_khadas_vim3)
        if(NOT ${_found_khadas_vim3} EQUAL -1)
            set(VIVANTE_SDK_FOR "A311D")
        else()
            string(FIND "{}" "Khadas VIM3L" _found_khadas_vim3l)
            if(NOT ${_found_khadas_vim3l} EQUAL -1)
                set(VIVANTE_SDK_FOR "S905D3")
            else()
                message("TIM-VX: Running on neither Khadas VIM3 nor VIM3L. TURNING OFF TIMVX_FOUND")
                set(TIMVX_FOUND OFF)
            endif()
        endif()

        list(INSERT vivante_sdk_filename 1 ${VIVANTE_SDK_FOR})
        list(JOIN vivante_sdk_filename "_" vivante_sdk_filename)

        set(VIVANTE_SDK_URL "https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.34.fix/")
        ocv_download(FILENAME ${vivante_sdk_filename}
                     HASH ${vivante_sdk_${VIVANTE_SDK_FOR}_md5sum}
                     URL "${VIVANTE_SDK_URL}"
                     DESTINATION_DIR "${OCV_VIVANTE_DIR}"
                     ID "${VIVANTE_SDK_FOR}"
                     STATUS res
                     UNPACK RELATIVE_URL)
        set(VIVANTE_SDK_DIR "${OCV_VIVANTE_DIR}/aarch64_${VIVANTE_SDK_FOR}_6.4.8")

        set(EXTERNAL_VIV_SDK "${VIVANTE_SDK_DIR}" CACHE INTERNAL "" FORCE)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
        set(VIVANTE_SDK_DIR "${OCV_TIMVX_SOURCE_PATH}/prebuilt-sdk/x86_64_linux")
        message(STATUS "TIM-VX: Build from source using prebuilt-sdk/x86_64_linux as VIVANTE_SDK_DIR")
    endif()

    set(TIMVX_INC_DIR "${OCV_TIMVX_SOURCE_PATH}/include" CACHE INTERNAL "TIM-VX include directory")
    if(EXISTS "${OCV_TIMVX_SOURCE_PATH}/CMakeLists.txt")
        add_subdirectory("${OCV_TIMVX_SOURCE_PATH}" "${OCV_TIMVX_DIR}/build")
    else()
        message(WARNING "TIM-VX: Missing 'CMakeLists.txt' in the source code: ${OCV_TIMVX_SOURCE_PATH}")
    endif()
    set(TIMVX_LIB "tim-vx")
endif()
