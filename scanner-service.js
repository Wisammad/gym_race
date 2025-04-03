const express = require('express');
const cors = require('cors');
const { execFile, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const app = express();
const port = 8080;

// Enable CORS
app.use(cors({
  origin: ['http://localhost:3000', 'http://192.168.68.103:3000'],
  methods: ['GET', 'POST'],
  credentials: true
}));

app.use(express.json());

// Detect OS platform
const isWindows = os.platform() === 'win32';
console.log(`Running on ${isWindows ? 'Windows' : 'Linux'} platform`);

// Setup paths to scanner application
const scannerPath = path.join(__dirname, 'samples', 'VP930Pro_bin');
const scannerExe = path.join(scannerPath, isWindows ? 'palm_test.exe' : 'palm_test');
const featuresDir = path.join(scannerPath, 'features');

// Ensure features directory exists
if (!fs.existsSync(featuresDir)) {
  fs.mkdirSync(featuresDir, { recursive: true });
}

// Convert feature text format to JSON structure
function parseFeatureFile(fileContent) {
  if (!fileContent) return null;
  
  const lines = fileContent.split('\n');
  const countMatch = lines[0].match(/Vector \((\d+) elements\)/);
  const count = countMatch ? parseInt(countMatch[1]) : 0;
  
  const features = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    const parts = line.split(':');
    if (parts.length !== 2) continue;
    
    const index = parseInt(parts[0]);
    const value = parseFloat(parts[1]);
    
    features.push(value);
  }
  
  return {
    count: count,
    values: features
  };
}

// Check if scanner executable exists
if (!fs.existsSync(scannerExe)) {
  console.error(`ERROR: Scanner executable not found at ${scannerExe}`);
  console.error(`Please ensure the scanner application is built for ${isWindows ? 'Windows' : 'Linux'}`);
  if (isWindows) {
    console.error('Run build_palm_test.bat to build for Windows');
  } else {
    console.error('Run ./build_palm_test.sh to build for Linux');
    console.error('Make sure to set execute permissions: chmod +x build_palm_test.sh');
  }
  process.exit(1);
}

// Check if required libraries exist
const requiredLibs = isWindows 
  ? ['palm_sdk.dll', 'palm-client-sdk.dll', 'dim_palm.dll', 'TNN.dll']
  : ['libpalm_sdk.so', 'libpalm-client-sdk.so', 'libdim_palm.so', 'libTNN.so'];

const missingLibs = requiredLibs.filter(lib => !fs.existsSync(path.join(scannerPath, lib)));
if (missingLibs.length > 0) {
  console.warn(`WARNING: Some required libraries are missing in ${scannerPath}:`);
  missingLibs.forEach(lib => console.warn(`  - ${lib}`));
  console.warn('The scanner may not function correctly.');
}

// Scanner interface
const scanner = {
  // Capture palm image and extract features
  capture: async () => {
    return new Promise((resolve, reject) => {
      console.log('Executing palm scanner...');
      
      // Use spawn instead of execFile to get real-time output
      const palmScanner = spawn(scannerExe, [], { 
        cwd: scannerPath,
        shell: true 
      });
      
      let output = '';
      let error = '';
      
      palmScanner.stdout.on('data', (data) => {
        const dataStr = data.toString();
        output += dataStr;
        console.log(`Scanner output: ${dataStr}`);
      });
      
      palmScanner.stderr.on('data', (data) => {
        const dataStr = data.toString();
        error += dataStr;
        console.error(`Scanner error: ${dataStr}`);
      });
      
      palmScanner.on('close', (code) => {
        console.log(`Scanner process exited with code ${code}`);
        
        if (code !== 0) {
          return reject(new Error(`Scanner process failed with code ${code}: ${error || output}`));
        }
        
        // Wait a moment to ensure files are fully written
        setTimeout(() => {
          try {
            // Look for latest feature files in the features directory
            const files = fs.readdirSync(featuresDir);
            const summaryFiles = files.filter(file => file.includes('palm_capture_summary'));
            
            if (summaryFiles.length === 0) {
              return reject(new Error('No feature files found after scanning'));
            }
            
            // Sort by timestamp descending
            summaryFiles.sort((a, b) => {
              const timestampA = a.split('_').pop().replace('.txt', '');
              const timestampB = b.split('_').pop().replace('.txt', '');
              return parseInt(timestampB) - parseInt(timestampA);
            });
            
            const latestSummary = summaryFiles[0];
            const timestamp = latestSummary.split('_').pop().replace('.txt', '');
            const irFeatureFile = path.join(featuresDir, `palm_features_ir_${timestamp}.txt`);
            const rgbFeatureFile = path.join(featuresDir, `palm_features_rgb_${timestamp}.txt`);
            const irBinFile = path.join(featuresDir, `palm_features_ir_${timestamp}.bin`);
            const rgbBinFile = path.join(featuresDir, `palm_features_rgb_${timestamp}.bin`);
            const summaryPath = path.join(featuresDir, latestSummary);
            
            // Read feature data
            const summary = fs.readFileSync(summaryPath, 'utf8');
            let irFeatures = null;
            let rgbFeatures = null;
            let irBinary = null;
            let rgbBinary = null;
            
            if (fs.existsSync(irFeatureFile)) {
              const irText = fs.readFileSync(irFeatureFile, 'utf8');
              irFeatures = parseFeatureFile(irText);
            }
            
            if (fs.existsSync(rgbFeatureFile)) {
              const rgbText = fs.readFileSync(rgbFeatureFile, 'utf8');
              rgbFeatures = parseFeatureFile(rgbText);
            }
            
            if (fs.existsSync(irBinFile)) {
              irBinary = fs.readFileSync(irBinFile).toString('base64');
            }
            
            if (fs.existsSync(rgbBinFile)) {
              rgbBinary = fs.readFileSync(rgbBinFile).toString('base64');
            }
            
            // Parse the summary data
            const qualityMatch = summary.match(/Quality Score: ([\d.]+)/);
            const palmTypeMatch = summary.match(/Palm Type: ([\d]+)/);
            const captureStatusMatch = summary.match(/Capture Status: ([\d]+)/);
            const liveErrorsMatch = summary.match(/Live Palm Errors: ([\d]+)/);
            
            // Check if the capture was successful
            const captureStatus = captureStatusMatch ? parseInt(captureStatusMatch[1]) : 1;
            const liveErrors = liveErrorsMatch ? parseInt(liveErrorsMatch[1]) : 1;
            const scanSuccessful = captureStatus === 0 && liveErrors === 0;
            
            if (!scanSuccessful) {
              return reject(new Error(`Palm scan unsuccessful: status=${captureStatus}, errors=${liveErrors}`));
            }
            
            resolve({
              success: true,
              summary: summary,
              irFeatures: irFeatures,
              rgbFeatures: rgbFeatures,
              irFeaturesBinary: irBinary,
              rgbFeaturesBinary: rgbBinary,
              quality: qualityMatch ? parseFloat(qualityMatch[1]) : 0,
              palmType: palmTypeMatch ? parseInt(palmTypeMatch[1]) : 0,
              timestamp: timestamp,
              captureStatus: captureStatus,
              liveErrors: liveErrors,
              metadata: {
                captureDevice: 'Palm Scanner VP930Pro',
                captureTime: new Date().toISOString(),
                format: 'PalmScannerSDK',
                platform: isWindows ? 'Windows' : 'Linux'
              }
            });
          } catch (err) {
            reject(new Error(`Error processing scanner results: ${err.message}`));
          }
        }, 500); // Wait 500ms to ensure files are fully written
      });
    });
  },
  
  // Get latest feature data
  getLatestFeatures: () => {
    try {
      const files = fs.readdirSync(featuresDir);
      const summaryFiles = files.filter(file => file.includes('palm_capture_summary'));
      
      if (summaryFiles.length === 0) {
        return null;
      }
      
      // Sort by timestamp descending
      summaryFiles.sort((a, b) => {
        const timestampA = a.split('_').pop().replace('.txt', '');
        const timestampB = b.split('_').pop().replace('.txt', '');
        return parseInt(timestampB) - parseInt(timestampA);
      });
      
      const latestSummary = summaryFiles[0];
      const timestamp = latestSummary.split('_').pop().replace('.txt', '');
      const irFeatureFile = path.join(featuresDir, `palm_features_ir_${timestamp}.txt`);
      const rgbFeatureFile = path.join(featuresDir, `palm_features_rgb_${timestamp}.txt`);
      const irBinFile = path.join(featuresDir, `palm_features_ir_${timestamp}.bin`);
      const rgbBinFile = path.join(featuresDir, `palm_features_rgb_${timestamp}.bin`);
      const summaryPath = path.join(featuresDir, latestSummary);
      
      // Read feature data
      const summary = fs.readFileSync(summaryPath, 'utf8');
      let irFeatures = null;
      let rgbFeatures = null;
      let irBinary = null;
      let rgbBinary = null;
      
      if (fs.existsSync(irFeatureFile)) {
        const irText = fs.readFileSync(irFeatureFile, 'utf8');
        irFeatures = parseFeatureFile(irText);
      }
      
      if (fs.existsSync(rgbFeatureFile)) {
        const rgbText = fs.readFileSync(rgbFeatureFile, 'utf8');
        rgbFeatures = parseFeatureFile(rgbText);
      }
      
      if (fs.existsSync(irBinFile)) {
        irBinary = fs.readFileSync(irBinFile).toString('base64');
      }
      
      if (fs.existsSync(rgbBinFile)) {
        rgbBinary = fs.readFileSync(rgbBinFile).toString('base64');
      }
      
      // Parse the summary data
      const qualityMatch = summary.match(/Quality Score: ([\d.]+)/);
      const palmTypeMatch = summary.match(/Palm Type: ([\d]+)/);
      const captureStatusMatch = summary.match(/Capture Status: ([\d]+)/);
      const liveErrorsMatch = summary.match(/Live Palm Errors: ([\d]+)/);
      
      return {
        summary: summary,
        irFeatures: irFeatures,
        rgbFeatures: rgbFeatures,
        irFeaturesBinary: irBinary,
        rgbFeaturesBinary: rgbBinary,
        quality: qualityMatch ? parseFloat(qualityMatch[1]) : 0,
        palmType: palmTypeMatch ? parseInt(palmTypeMatch[1]) : 0,
        timestamp: timestamp,
        captureStatus: captureStatusMatch ? parseInt(captureStatusMatch[1]) : 1,
        liveErrors: liveErrorsMatch ? parseInt(liveErrorsMatch[1]) : 1,
        metadata: {
          captureDevice: 'Palm Scanner VP930Pro',
          captureTime: new Date().toISOString(),
          format: 'PalmScannerSDK'
        }
      };
    } catch (error) {
      console.error('Error getting latest features:', error);
      return null;
    }
  },
  
  // List all available feature sets
  listFeatureSets: () => {
    try {
      const files = fs.readdirSync(featuresDir);
      const summaryFiles = files.filter(file => file.includes('palm_capture_summary'));
      
      return summaryFiles.map(file => {
        const timestamp = file.split('_').pop().replace('.txt', '');
        const summaryPath = path.join(featuresDir, file);
        const summary = fs.readFileSync(summaryPath, 'utf8');
        
        // Parse the summary data
        const qualityMatch = summary.match(/Quality Score: ([\d.]+)/);
        const palmTypeMatch = summary.match(/Palm Type: ([\d]+)/);
        const captureStatusMatch = summary.match(/Capture Status: ([\d]+)/);
        const liveErrorsMatch = summary.match(/Live Palm Errors: ([\d]+)/);
        
        return {
          timestamp: timestamp,
          quality: qualityMatch ? parseFloat(qualityMatch[1]) : 0,
          palmType: palmTypeMatch ? parseInt(palmTypeMatch[1]) : 0,
          captureStatus: captureStatusMatch ? parseInt(captureStatusMatch[1]) : 1,
          liveErrors: liveErrorsMatch ? parseInt(liveErrorsMatch[1]) : 1,
          captureTime: new Date(parseInt(timestamp)).toISOString()
        };
      }).sort((a, b) => parseInt(b.timestamp) - parseInt(a.timestamp));
    } catch (error) {
      console.error('Error listing feature sets:', error);
      return [];
    }
  }
};

// Health check endpoint
app.get('/health', (req, res) => {
  const scannerExists = fs.existsSync(scannerExe);
  const featuresPathExists = fs.existsSync(featuresDir);
  
  res.json({ 
    status: 'ok', 
    service: 'Palm Scanner API',
    scanner: {
      executable: scannerExe,
      exists: scannerExists,
      featuresDir: featuresDir,
      featuresPathExists: featuresPathExists
    }
  });
});

// Capture palm endpoint
app.post('/capture', async (req, res) => {
  try {
    const result = await scanner.capture();
    res.json(result);
  } catch (error) {
    console.error('Capture error:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message,
      details: error.stack
    });
  }
});

// Get latest features endpoint
app.get('/features/latest', (req, res) => {
  try {
    const features = scanner.getLatestFeatures();
    if (!features) {
      return res.status(404).json({ success: false, error: 'No feature data found' });
    }
    res.json({ success: true, data: features });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// List all available feature sets
app.get('/features/list', (req, res) => {
  try {
    const featureSets = scanner.listFeatureSets();
    res.json({ success: true, data: featureSets });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get specific feature set by timestamp
app.get('/features/:timestamp', (req, res) => {
  try {
    const { timestamp } = req.params;
    const irFeatureFile = path.join(featuresDir, `palm_features_ir_${timestamp}.txt`);
    const rgbFeatureFile = path.join(featuresDir, `palm_features_rgb_${timestamp}.txt`);
    const irBinFile = path.join(featuresDir, `palm_features_ir_${timestamp}.bin`);
    const rgbBinFile = path.join(featuresDir, `palm_features_rgb_${timestamp}.bin`);
    const summaryPath = path.join(featuresDir, `palm_capture_summary_${timestamp}.txt`);
    
    if (!fs.existsSync(summaryPath)) {
      return res.status(404).json({ success: false, error: 'Feature set not found' });
    }
    
    // Read feature data
    const summary = fs.readFileSync(summaryPath, 'utf8');
    let irFeatures = null;
    let rgbFeatures = null;
    let irBinary = null;
    let rgbBinary = null;
    
    if (fs.existsSync(irFeatureFile)) {
      const irText = fs.readFileSync(irFeatureFile, 'utf8');
      irFeatures = parseFeatureFile(irText);
    }
    
    if (fs.existsSync(rgbFeatureFile)) {
      const rgbText = fs.readFileSync(rgbFeatureFile, 'utf8');
      rgbFeatures = parseFeatureFile(rgbText);
    }
    
    if (fs.existsSync(irBinFile)) {
      irBinary = fs.readFileSync(irBinFile).toString('base64');
    }
    
    if (fs.existsSync(rgbBinFile)) {
      rgbBinary = fs.readFileSync(rgbBinFile).toString('base64');
    }
    
    // Parse the summary data
    const qualityMatch = summary.match(/Quality Score: ([\d.]+)/);
    const palmTypeMatch = summary.match(/Palm Type: ([\d]+)/);
    
    res.json({ 
      success: true, 
      data: {
        summary: summary,
        irFeatures: irFeatures,
        rgbFeatures: rgbFeatures,
        irFeaturesBinary: irBinary,
        rgbFeaturesBinary: rgbBinary,
        quality: qualityMatch ? parseFloat(qualityMatch[1]) : 0,
        palmType: palmTypeMatch ? parseInt(palmTypeMatch[1]) : 0,
        timestamp: timestamp
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.listen(port, '0.0.0.0', () => {
  console.log(`======================================================`);
  console.log(`Palm Scanner API service running on http://0.0.0.0:${port}`);
  console.log(`------------------------------------------------------`);
  console.log(`Platform: ${isWindows ? 'Windows' : 'Linux'}`);
  console.log(`Scanner path: ${scannerPath}`);
  console.log(`Scanner executable: ${scannerExe}`);
  console.log(`Features directory: ${featuresDir}`);
  console.log(`======================================================`);
  
  // List available endpoints
  console.log(`Available endpoints:`);
  console.log(`- GET  /health`);
  console.log(`- POST /capture`);
  console.log(`- GET  /features/latest`);
  console.log(`- GET  /features/list`);
  console.log(`- GET  /features/:timestamp`);
  console.log(`======================================================`);
}); 