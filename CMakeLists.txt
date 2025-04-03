cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(palm_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)  # Changed to C++17 for <filesystem>
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if we're building for Linux
option(BUILD_FOR_LINUX "Build for Linux platform" OFF)
option(BUILD_FOR_ARM64 "Build for ARM64 architecture (Raspberry Pi 5)" OFF)

# Platform-specific settings
if(BUILD_FOR_LINUX)
  # Linux-specific settings
  message(STATUS "Building for Linux platform")
  add_definitions(-DBUILD_FOR_LINUX)
  
  # Add pthread if required
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  
  # ARM64-specific settings
  if(BUILD_FOR_ARM64)
    message(STATUS "Building for ARM64 architecture (Raspberry Pi 5)")
    add_definitions(-DBUILD_FOR_ARM64)
    
    # ARM64-specific compiler flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    
    # Use ARM NEON instructions if available
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-mfpu=neon" COMPILER_SUPPORTS_NEON)
    if(COMPILER_SUPPORTS_NEON)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    endif()
  endif()
else()
  # Windows-specific settings
  message(STATUS "Building for Windows platform")
endif()

# Disable interface by default since we don't have image_viewer
add_definitions(-DDISABLE_INTERFACE)

# Include directories - corrected paths based on working example
include_directories(.)
include_directories(..)  # Added back to find frame_rate_helper.h
include_directories(../../../include)

# Add arch-specific library directories
if(BUILD_FOR_LINUX)
  if(BUILD_FOR_ARM64)
    link_directories(../../../lib/arm64)
  else()
    link_directories(../../../lib)
  endif()
else()
  link_directories(../../../lib)
endif()

# Find OpenCV
if(NOT DEFINED OpenCV_DIR)
  if(BUILD_FOR_LINUX)
    # Set OpenCV_DIR for Linux
    # Use find_package without a specific path for Linux
    if(BUILD_FOR_ARM64)
      # For Raspberry Pi, OpenCV is typically installed in system directories
      # Check if installed with apt or manual build
      if(EXISTS "/usr/local/lib/arm-linux-gnueabihf")
        set(OpenCV_DIR "/usr/local/lib/arm-linux-gnueabihf/cmake/opencv4")
      endif()
    endif()
  else()
    # Set OpenCV_DIR to the downloaded OpenCV location for Windows
    set(OpenCV_DIR "C:/Users/Wisam/Downloads/opencv/build/x64/vc16/lib")
  endif()
endif()

find_package(OpenCV REQUIRED)
if(DEFINED OpenCV_LIBS AND DEFINED OpenCV_INCLUDE_DIRS)
  set(OpenCV_FOUND TRUE)
  include_directories(${OpenCV_INCLUDE_DIRS})
else()
  set(OpenCV_FOUND FALSE)
endif()

# Main palm_test files
set(SAMPLE_FILES
    palm_device.h
    palm_device.cc
    palm_test.cc
)

set(SAMPLE_COMMON_FILES
    ../frame_rate_helper.h 
    ../sample_utils.h 
    ../sample_utils.cc
)

# Main executable
add_executable(palm_test ${SAMPLE_FILES} ${SAMPLE_COMMON_FILES})

# Platform-specific linking
if(BUILD_FOR_LINUX)
  target_link_libraries(palm_test
    ${OpenCV_LIBS}
    palm_sdk
    stdc++fs  # For filesystem library on Linux
  )
else()
  target_link_libraries(palm_test
    ${OpenCV_LIBS}
    palm_sdk
  )
endif()

# Install targets
if(BUILD_FOR_LINUX)
  install(TARGETS palm_test DESTINATION samples/VP930Pro_bin)
else()
  install(TARGETS palm_test DESTINATION samples/VP930Pro_bin)
endif()


