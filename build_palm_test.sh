#!/bin/bash
echo "Building Palm Test Application for Linux (ARM64/Raspberry Pi 5 compatible)"

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Clean existing builds
if [ -d "build_linux" ]; then
  rm -rf build_linux
fi

# Create build directory
mkdir -p build_linux

# Copy config files to build directory
echo
echo "===== Copying config files ====="
mkdir -p build_linux/config
cp -r samples/VP930Pro_bin/config/* build_linux/config/

# Build for Linux
echo
echo "===== Building for Linux (ARM64 compatible) ====="
cd build_linux

# ARM specific flags if building on ARM
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  echo "Setting ARM64 specific compiler flags"
  export CFLAGS="-march=armv8-a -mtune=cortex-a72"
  export CXXFLAGS="-march=armv8-a -mtune=cortex-a72"
  cmake -DCMAKE_INSTALL_PREFIX=../ -DBUILD_FOR_LINUX=ON -DBUILD_FOR_ARM64=ON ../samples/src/sample
else
  echo "Building on non-ARM platform, not setting ARM-specific flags"
  cmake -DCMAKE_INSTALL_PREFIX=../ -DBUILD_FOR_LINUX=ON ../samples/src/sample
fi

cmake --build . --config Release
cd ..

# Copy shared libraries and config files
echo
echo "===== Copying shared libraries and executables to target directory ====="
mkdir -p samples/VP930Pro_bin
cp -f build_linux/palm_test samples/VP930Pro_bin/palm_test

# Copy the shared libraries to the executable directory if not already there
if [ ! -f samples/VP930Pro_bin/libpalm_sdk.so ]; then
  echo "Copying palm SDK libraries to target directory..."
  
  # Copy appropriate architecture libraries
  if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    echo "Copying ARM64 libraries"
    cp -f lib/arm64/*.so samples/VP930Pro_bin/ 2>/dev/null
  else
    echo "Copying standard libraries"
    cp -f lib/*.so samples/VP930Pro_bin/ 2>/dev/null
  fi
fi

# Create config directory in build_linux folder as well
mkdir -p build_linux/config
cp -r samples/VP930Pro_bin/config/* build_linux/config/

echo
echo "Build completed successfully!"
echo
echo "To run the application, use:"
echo "  cd samples/VP930Pro_bin"
echo "  ./palm_test" 