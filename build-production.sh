#!/bin/bash

# IAA Feedback System - Production Build Script

echo "🚀 Building IAA Feedback System for Production..."

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

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed"
        exit 1
    fi
    
    if ! command -v cargo &> /dev/null; then
        print_error "Rust/Cargo is not installed"
        exit 1
    fi
    
    print_success "All dependencies are available"
}

# Build frontend
build_frontend() {
    print_status "Building React frontend..."
    
    cd iaa-feedback-system
    
    # Install dependencies
    print_status "Installing frontend dependencies..."
    npm ci
    
    # Build for production
    print_status "Building frontend for production..."
    npm run build
    
    if [ $? -eq 0 ]; then
        print_success "Frontend build completed successfully"
    else
        print_error "Frontend build failed"
        exit 1
    fi
    
    cd ..
}

# Build backend
build_backend() {
    print_status "Building Rust backend..."
    
    cd iaa-backend
    
    # Build for production
    print_status "Building backend for production..."
    cargo build --release --bin iaa-backend
    
    if [ $? -eq 0 ]; then
        print_success "Backend build completed successfully"
    else
        print_error "Backend build failed"
        exit 1
    fi
    
    cd ..
}

# Create production package
create_package() {
    print_status "Creating production package..."
    
    # Create production directory
    mkdir -p production-build
    
    # Copy backend binary
    cp iaa-backend/target/release/iaa-backend production-build/
    cp -r iaa-backend/migrations production-build/
    
    # Copy frontend build
    cp -r iaa-feedback-system/dist production-build/frontend
    
    # Copy configuration files
    cp .env.production production-build/
    cp docker-compose.production.yml production-build/
    cp DEPLOYMENT_GUIDE.md production-build/
    
    print_success "Production package created in 'production-build' directory"
}

# Main execution
main() {
    echo "=============================================="
    echo "🛩️  IAA Feedback System Production Build"
    echo "=============================================="
    
    check_dependencies
    build_frontend
    build_backend
    create_package
    
    echo ""
    echo "=============================================="
    print_success "🎉 Production build completed successfully!"
    echo "=============================================="
    echo ""
    echo "📁 Production files are in: ./production-build/"
    echo "📖 Deployment guide: ./production-build/DEPLOYMENT_GUIDE.md"
    echo ""
    echo "🚀 Your IAA Feedback System is ready for deployment!"
    echo ""
}

# Run main function
main
