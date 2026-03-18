cmake_minimum_required(VERSION 3.22.1)
project(yolo_ncnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED NCNN_ZIP_PATH)
    set(NCNN_ZIP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../ncnn-20260113-android-vulkan-shared.zip)
endif()

set(NCNN_EXTRACT_DIR ${CMAKE_BINARY_DIR}/ncnn_sdk)
set(NCNN_ROOT ${NCNN_EXTRACT_DIR}/ncnn-20260113-android-vulkan-shared)

if(NOT EXISTS ${NCNN_ROOT}/${ANDROID_ABI}/include)
    if(NOT EXISTS ${NCNN_ZIP_PATH})
        message(FATAL_ERROR "ncnn zip not found: ${NCNN_ZIP_PATH}")
    endif()
    file(ARCHIVE_EXTRACT INPUT ${NCNN_ZIP_PATH} DESTINATION ${NCNN_EXTRACT_DIR})
endif()

if (NOT EXISTS ${NCNN_ROOT}/${ANDROID_ABI}/include)
    message(FATAL_ERROR "ncnn include dir not found for ABI: ${ANDROID_ABI}")
endif()

if (NOT EXISTS ${NCNN_ROOT}/${ANDROID_ABI}/lib/libncnn.so)
    message(FATAL_ERROR "libncnn.so not found for ABI: ${ANDROID_ABI}")
endif()

add_library(ncnn SHARED IMPORTED)
set_target_properties(ncnn PROPERTIES
    IMPORTED_LOCATION ${NCNN_ROOT}/${ANDROID_ABI}/lib/libncnn.so
)

add_library(yolo_ncnn SHARED yolo_ncnn_jni.cpp)

target_include_directories(yolo_ncnn PRIVATE
    ${NCNN_ROOT}/${ANDROID_ABI}/include
)

find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(yolo_ncnn
    PRIVATE
    ncnn
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
)
