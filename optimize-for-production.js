#!/usr/bin/env node

/**
 * Production Optimization Script
 * Removes demo content, console logs, and optimizes for deployment
 */

const fs = require('fs');
const path = require('path');

console.log('🚀 Starting production optimization...');

// Files to process
const filesToOptimize = [
  'iaa-feedback-system/src/pages/AdminDashboard.jsx',
  'iaa-feedback-system/src/pages/TrainerDashboard.jsx',
  'iaa-feedback-system/src/pages/TraineeDashboard.jsx',
  'iaa-feedback-system/src/pages/RegisterPage.jsx',
  'iaa-feedback-system/src/pages/LoginPage.jsx',
  'iaa-feedback-system/src/api/authService.js',
  'iaa-feedback-system/src/components/NotificationSystem.jsx'
];

// Patterns to remove
const patternsToRemove = [
  /console\.(log|error|warn|info|debug)\([^)]*\);?\s*\n?/g,
  /\/\/ Demo.*\n/g,
  /\/\* Demo.*?\*\//gs,
  /\/\/ TODO.*\n/g,
  /\/\/ FIXME.*\n/g,
  /\/\/ DEBUG.*\n/g
];

// Demo content patterns
const demoPatterns = [
  /demo123@iaa\.edu\.in/g,
  /admin123@iaa\.edu\.in/g,
  /trainer123@iaa\.edu\.in/g,
  /trainee123@iaa\.edu\.in/g,
  /test456@iaa\.edu\.in/g,
  /student789@iaa\.edu\.in/g,
  /"admin123"/g,
  /"trainer123"/g,
  /"trainee123"/g,
  /"test123"/g,
  /"password"/g,
  /"student123"/g
];

function optimizeFile(filePath) {
  if (!fs.existsSync(filePath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return;
  }

  let content = fs.readFileSync(filePath, 'utf8');
  let originalLength = content.length;

  // Remove console logs and debug statements
  patternsToRemove.forEach(pattern => {
    content = content.replace(pattern, '');
  });

  // Remove demo credentials (replace with placeholders)
  demoPatterns.forEach(pattern => {
    content = content.replace(pattern, '"[REMOVED_FOR_PRODUCTION]"');
  });

  // Remove empty lines (more than 2 consecutive)
  content = content.replace(/\n\s*\n\s*\n/g, '\n\n');

  // Remove trailing whitespace
  content = content.replace(/[ \t]+$/gm, '');

  let newLength = content.length;
  let reduction = ((originalLength - newLength) / originalLength * 100).toFixed(1);

  fs.writeFileSync(filePath, content);
  console.log(`✅ Optimized ${filePath} (${reduction}% reduction)`);
}

function createProductionConfig() {
  const packageJsonPath = 'iaa-feedback-system/package.json';
  if (fs.existsSync(packageJsonPath)) {
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    
    // Add production build script
    packageJson.scripts = packageJson.scripts || {};
    packageJson.scripts['build:prod'] = 'NODE_ENV=production npm run build';
    packageJson.scripts['analyze'] = 'npm run build && npx webpack-bundle-analyzer build/static/js/*.js';
    
    fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
    console.log('✅ Updated package.json with production scripts');
  }
}

function removeTestFiles() {
  const testFiles = [
    'iaa-feedback-system/src/TestApp.jsx',
    'iaa-feedback-system/src/components/BackendTest.jsx'
  ];

  testFiles.forEach(file => {
    if (fs.existsSync(file)) {
      fs.unlinkSync(file);
      console.log(`🗑️  Removed test file: ${file}`);
    }
  });
}

function optimizeImages() {
  const imagesDir = 'iaa-feedback-system/public/images';
  if (fs.existsSync(imagesDir)) {
    console.log('📸 Image optimization would be performed here');
    // In a real scenario, you'd use imagemin or similar
  }
}

function generateBuildReport() {
  const report = {
    timestamp: new Date().toISOString(),
    optimizations: [
      'Removed console.log statements',
      'Removed demo credentials',
      'Removed test files',
      'Optimized whitespace',
      'Added production configuration'
    ],
    filesOptimized: filesToOptimize.length,
    environment: 'production'
  };

  fs.writeFileSync('production-optimization-report.json', JSON.stringify(report, null, 2));
  console.log('📊 Generated optimization report');
}

// Main execution
function main() {
  console.log('🛩️  IAA Feedback System - Production Optimization');
  console.log('================================================');

  // Optimize source files
  filesToOptimize.forEach(optimizeFile);

  // Create production configuration
  createProductionConfig();

  // Remove test files
  removeTestFiles();

  // Optimize images (placeholder)
  optimizeImages();

  // Generate report
  generateBuildReport();

  console.log('================================================');
  console.log('✅ Production optimization completed!');
  console.log('');
  console.log('Next steps:');
  console.log('1. Run: npm run build:prod');
  console.log('2. Test the production build');
  console.log('3. Deploy to your hosting platform');
  console.log('');
  console.log('🚀 Your IAA Feedback System is ready for production!');
}

if (require.main === module) {
  main();
}

module.exports = { optimizeFile, main };
