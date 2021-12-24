set(TIMVX_INSTALL_DIR "" CACHE PATH "Path to libtim-vx installation")
set(VIVANTE_SDK_DIR "" CACHE PATH "Path to VIVANTE SDK needed by TIM-VX.")
set(VIVANTE_SDK_LIB_CANDIDATES "ArchModelSw;CLC;Emulator;GAL;NNArchPerf;NNGPUBinary;NNVXCBinary;Ovx12VXCBinary;OpenVXC;OpenVX;OpenVXU;vdtproxy;VSC" CACHE STRING "TIM-VX's SDK library candidates list")

function(find_vivante_sdk_libs _found)
    foreach(one ${VIVANTE_SDK_LIB_CANDIDATES})
        find_library(TIMVX_${one}_LIBRARY ${one} PATHS "${VIVANTE_SDK_DIR}/lib" NO_DEFAULT_PATH) # exclude /lib etc. and use the specified VIVANTE_SDK_DIR only
        if(TIMVX_${one}_LIBRARY)
            list(APPEND _sdk_libs ${TIMVX_${one}_LIBRARY})
        endif()
    endforeach()
    if(EXISTS "${VIVANTE_SDK_DIR}/lib/libOpenVX.so.1")
        list(APPEND _sdk_libs ${VIVANTE_SDK_DIR}/lib/libOpenVX.so.1)
    endif()
    if(EXISTS "${VIVANTE_SDK_DIR}/lib/libOpenVX.so.1.3.0")
        list(APPEND _sdk_libs ${VIVANTE_SDK_DIR}/lib/libOpenVX.so.1.3.0)
    endif()
    set(${_found} ${_sdk_libs} PARENT_SCOPE)
endfunction()

if(TIMVX_INSTALL_DIR AND NOT BUILD_TIMVX)
    message(STATUS "TIM-VX: Use binaries at ${TIMVX_INSTALL_DIR}")

    set(TIMVX_FOUND ON)
    set(BUILD_TIMVX OFF)

    set(TIMVX_INC_DIR "${TIMVX_INSTALL_DIR}/include" CACHE PATH "TIM-VX include directory")
    find_library(TIMVX_LIB "tim-vx" PATHS "${TIMVX_INSTALL_DIR}/lib")
    #message(STATUS ${TIMVX_LIB})

    if(VIVANTE_SDK_DIR)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,\"${VIVANTE_SDK_DIR}/lib\"")
        find_vivante_sdk_libs(VIVANTE_SDK_LIBS)
    else()
        message(STATUS "TIM-VX: Error: VIVANTE_SDK_DIR is required to build with TIM-VX. Turning off TIMVX_FOUND")
        set(TIMVX_FOUND OFF)
        return()
    endif()

    try_compile(VALID_TIMVX
        "${OpenCV_BINARY_DIR}"
        "${OpenCV_SOURCE_DIR}/cmake/checks/tim-vx.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${TIMVX_INC_DIR};${VIVANTE_SDK_LIBS}"
        LINK_LIBRARIES ${TIMVX_LIB};${VIVANTE_SDK_LIBS}
        OUTPUT_VARIABLE OUTPUT
    )
    if(NOT VALID_TIMVX)
        message(WARNING "TIM-VX: Failed to compile ${OpenCV_SOURCE_DIR}/cmake/checks/tim-vx.cpp. Turning off TIMVX_FOUND")
        set(TIMVX_FOUND OFF)
        return()
    endif()

else()
    message(STATUS "TIM-VX: Build from source")
    # build with TIM-VX x86-64 SDK
    include("${OpenCV_SOURCE_DIR}/3rdparty/libtim-vx/tim-vx.cmake")

    if(VIVANTE_SDK_DIR) # VIVANTE_SDK_DIR will be given in tim-vx.cmake
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,\"${VIVANTE_SDK_DIR}/lib\"")
        find_vivante_sdk_libs(VIVANTE_SDK_LIBS)
    endif()

    if(NOT TIMVX_LIB)
        set(TIMVX_FOUND OFF)
        message(STATUS "TIM-VX: Failed to find TIM-VX library. Turning off TIMVX_FOUND")
        return()
    endif()
endif()

if(TIMVX_FOUND)
    set(HAVE_TIMVX 1)

    message(STATUS "TIM-VX: Found TIM-VX includes: ${TIMVX_INC_DIR}")
    message(STATUS "TIM-VX: Found TIM-VX library: ${TIMVX_LIB}")
    set(TIMVX_LIBRARY   ${TIMVX_LIB})
    set(TIMVX_INCLUDE_DIR   ${TIMVX_INC_DIR})

    message(STATUS "TIM-VX: Found VIVANTE SDK includes for TIM-VX: ${VIVANTE_SDK_DIR}/include")
    message(STATUS "TIM-VX: Found VIVANTE SDK libraries for TIM-VX: ${VIVANTE_SDK_LIBS}")
    set(VIVANTE_SDK_LIBRARIES ${VIVANTE_SDK_LIBS})
    set(VIVANTE_SDK_INCLUDE_DIR ${VIVANTE_SDK_DIR}/include)

    install(FILES ${VIVANTE_SDK_DIR}/include/CL/cl_viv_vx_ext.h DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/CL COMPONENT vivante)
    foreach(one ${VIVANTE_SDK_LIBS})
        install(FILES ${one} DESTINATION ${CMAKE_INSTALL_PREFIX}/${OPENCV_LIB_INSTALL_PATH} COMPONENT vivante)
    endforeach()

endif()

MARK_AS_ADVANCED(
	TIMVX_INC_DIR
	TIMVX_LIB
    VIVANTE_SDK_LIBS
)
