#!/bin/bash
echo "==================================================="
echo "Starting Palm Scanner API Service..."
echo "==================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH."
    echo "Please install Node.js from https://nodejs.org/"
    echo "==================================================="
    exit 1
fi

# Check if npm modules are installed
if [ ! -d "node_modules" ]; then
    echo "First-time setup: Installing required npm packages..."
    npm install
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install required npm packages."
        echo "==================================================="
        exit 1
    fi
fi

# Ensure features directory exists
mkdir -p samples/VP930Pro_bin/features

# Start the service
echo
echo "Starting server on http://localhost:8080"
echo "Press Ctrl+C to stop the service"
echo "==================================================="
node scanner-service.js 