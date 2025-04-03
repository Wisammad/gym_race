#!/bin/bash
echo "Building Palm Test Application for Linux"

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
echo "===== Building for Linux ====="
cd build_linux
cmake -DCMAKE_INSTALL_PREFIX=../ -DBUILD_FOR_LINUX=ON ../samples/src/sample
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
  cp -f lib/*.so samples/VP930Pro_bin/ 2>/dev/null
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