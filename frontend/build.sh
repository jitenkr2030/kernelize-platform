#!/bin/bash

# KERNELIZE Platform Frontend Build Script
# This script builds the frontend and prepares it for deployment

set -e

echo "🚀 Building KERNELIZE Platform Frontend..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ to continue."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    print_error "Node.js version 18+ is required. Current version: $(node -v)"
    exit 1
fi

print_success "Node.js version check passed: $(node -v)"

# Change to frontend directory
cd "$(dirname "$0")"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    print_error "package.json not found in frontend directory"
    exit 1
fi

# Install dependencies
print_status "Installing dependencies..."
if command -v yarn &> /dev/null; then
    yarn install --frozen-lockfile
else
    npm ci
fi

print_success "Dependencies installed successfully"

# Run linting
print_status "Running linting..."
if command -v yarn &> /dev/null; then
    yarn lint
else
    npm run lint
fi

# Run type checking
print_status "Running TypeScript type checking..."
if command -v yarn &> /dev/null; then
    yarn type-check
else
    npm run type-check
fi

# Clean previous build
print_status "Cleaning previous build..."
rm -rf dist/
rm -rf node_modules/.vite/

# Build the application
print_status "Building application for production..."
if command -v yarn &> /dev/null; then
    yarn build
else
    npm run build
fi

# Check if build was successful
if [ -d "dist" ]; then
    print_success "Build completed successfully!"
    
    # Display build information
    BUILD_SIZE=$(du -sh dist/ | cut -f1)
    print_status "Build size: $BUILD_SIZE"
    
    # List key files
    print_status "Key files in build:"
    ls -la dist/ | head -10
    
    # Create a simple HTTP server for testing
    print_status "Starting local server for testing..."
    print_warning "Press Ctrl+C to stop the server"
    
    if command -v python3 &> /dev/null; then
        python3 -m http.server 8080 --directory dist/
    elif command -v python &> /dev/null; then
        python -m SimpleHTTPServer 8080 --directory dist/
    elif command -v npx &> /dev/null; then
        npx serve dist/ -l 8080
    else
        print_warning "No HTTP server available. You can manually serve the dist/ folder."
        print_status "Build is ready in the 'dist' directory"
    fi
    
else
    print_error "Build failed! Check the error messages above."
    exit 1
fi

print_success "🎉 KERNELIZE Platform Frontend build completed!"