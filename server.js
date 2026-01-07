const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for memory storage
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 5 * 1024 * 1024, // 5MB limit
    },
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed!'), false);
        }
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'OK',
        message: 'Emotion Detection API with Age Recognition is running',
        timestamp: new Date().toISOString(),
        features: ['emotion_detection', 'age_detection', 'music_recommendations']
    });
});

// UNIFIED endpoint for emotion + age detection + music recommendations
app.post('/predict-with-recommendations', upload.single('image'), async (req, res) => {
    let tempImagePath = null;

    try {
        console.log('🎵 Received image for emotion + age detection + music recommendations');

        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        console.log('🖼️ Image details:', {
            size: req.file.size,
            mimetype: req.file.mimetype,
            originalname: req.file.originalname
        });

        // Create temp directory if it doesn't exist
        const tempDir = path.join(__dirname, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir);
        }

        // Save image to temporary file
        const timestamp = Date.now();
        tempImagePath = path.join(tempDir, `temp_image_${timestamp}.jpg`);
        fs.writeFileSync(tempImagePath, req.file.buffer);
        console.log('💾 Image saved to temp file:', tempImagePath);

        // Call predict_emotion.py which handles BOTH emotion and age
        const pythonProcess = spawn('python', ['predict_emotion.py', tempImagePath]);

        let resultData = '';
        let errorData = '';

        pythonProcess.stdout.on('data', (data) => {
            resultData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorData += data.toString();
            console.error('🐍 Python stderr:', data.toString());
        });

        pythonProcess.on('close', (code) => {
            // Clean up temp file
            try {
                if (tempImagePath && fs.existsSync(tempImagePath)) {
                    fs.unlinkSync(tempImagePath);
                    console.log('🗑️ Temp file cleaned up');
                }
            } catch (cleanupError) {
                console.warn('⚠️ Warning: Could not delete temp file:', cleanupError.message);
            }

            console.log(`🐍 Python process exited with code ${code}`);

            if (code !== 0) {
                console.error('❌ Python process failed with code:', code);
                console.error('❌ Error output:', errorData);
                return res.status(500).json({
                    error: 'Prediction failed',
                    details: errorData || 'Python process exited with non-zero code'
                });
            }

            if (!resultData.trim()) {
                console.error('❌ No output from Python script');
                return res.status(500).json({
                    error: 'No output from prediction model'
                });
            }

            try {
                // Find the JSON result in the output (last line should be JSON)
                const lines = resultData.trim().split('\n');
                const jsonLine = lines[lines.length - 1];

                const prediction = JSON.parse(jsonLine);

                if (prediction.status === 'error') {
                    console.error('❌ Python script returned error:', prediction.error);
                    return res.status(500).json(prediction);
                }

                console.log('✅ Prediction successful:', {
                    age_category: prediction.age_category,
                    emotion: prediction.emotion,
                    is_child: prediction.is_child,
                    recommendations: prediction.music_recommendations?.length || 0
                });

                res.json(prediction);
            } catch (parseError) {
                console.error('❌ JSON parse error:', parseError.message);
                console.error('❌ Raw Python output:', resultData);
                res.status(500).json({
                    error: 'Failed to parse prediction result',
                    details: parseError.message,
                    raw_output: resultData
                });
            }
        });

        pythonProcess.on('error', (error) => {
            console.error('💥 Python process spawn error:', error);
            // Clean up temp file on error
            try {
                if (tempImagePath && fs.existsSync(tempImagePath)) {
                    fs.unlinkSync(tempImagePath);
                }
            } catch (cleanupError) {
                console.warn('Warning: Could not delete temp file:', cleanupError.message);
            }

            res.status(500).json({
                error: 'Failed to start prediction process',
                details: error.message
            });
        });

    } catch (error) {
        console.error('💥 Server error:', error);

        // Clean up temp file on error
        try {
            if (tempImagePath && fs.existsSync(tempImagePath)) {
                fs.unlinkSync(tempImagePath);
            }
        } catch (cleanupError) {
            console.warn('Warning: Could not delete temp file:', cleanupError.message);
        }

        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Backward compatible endpoint (emotion only)
app.post('/predict-emotion', upload.single('image'), async (req, res) => {
    let tempImagePath = null;

    try {
        console.log('📸 Received image for emotion detection');

        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        console.log('🖼️ Image details:', {
            size: req.file.size,
            mimetype: req.file.mimetype,
            originalname: req.file.originalname
        });

        // Create temp directory if it doesn't exist
        const tempDir = path.join(__dirname, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir);
        }

        // Save image to temporary file
        const timestamp = Date.now();
        tempImagePath = path.join(tempDir, `temp_image_${timestamp}.jpg`);
        fs.writeFileSync(tempImagePath, req.file.buffer);
        console.log('💾 Image saved to temp file:', tempImagePath);

        // Call Python script with file path
        const pythonProcess = spawn('python', ['predict_emotion.py', tempImagePath]);

        let resultData = '';
        let errorData = '';

        pythonProcess.stdout.on('data', (data) => {
            resultData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorData += data.toString();
            console.error('🐍 Python stderr:', data.toString());
        });

        pythonProcess.on('close', (code) => {
            // Clean up temp file
            try {
                if (tempImagePath && fs.existsSync(tempImagePath)) {
                    fs.unlinkSync(tempImagePath);
                    console.log('🗑️ Temp file cleaned up');
                }
            } catch (cleanupError) {
                console.warn('⚠️ Warning: Could not delete temp file:', cleanupError.message);
            }

            console.log(`🐍 Python process exited with code ${code}`);

            if (code !== 0) {
                console.error('❌ Python process failed with code:', code);
                console.error('❌ Error output:', errorData);
                return res.status(500).json({
                    error: 'Emotion prediction failed',
                    details: errorData || 'Python process exited with non-zero code'
                });
            }

            if (!resultData.trim()) {
                console.error('❌ No output from Python script');
                return res.status(500).json({
                    error: 'No output from emotion detection model'
                });
            }

            try {
                // Find the JSON result in the output (last line should be JSON)
                const lines = resultData.trim().split('\n');
                const jsonLine = lines[lines.length - 1];

                const prediction = JSON.parse(jsonLine);

                if (prediction.status === 'error') {
                    console.error('❌ Python script returned error:', prediction.error);
                    return res.status(500).json(prediction);
                }

                console.log('✅ Prediction successful:', {
                    emotion: prediction.emotion,
                    confidence: prediction.emotion_confidence
                });

                res.json(prediction);
            } catch (parseError) {
                console.error('❌ JSON parse error:', parseError.message);
                console.error('❌ Raw Python output:', resultData);
                res.status(500).json({
                    error: 'Failed to parse prediction result',
                    details: parseError.message,
                    raw_output: resultData
                });
            }
        });

        pythonProcess.on('error', (error) => {
            console.error('💥 Python process spawn error:', error);
            // Clean up temp file on error
            try {
                if (tempImagePath && fs.existsSync(tempImagePath)) {
                    fs.unlinkSync(tempImagePath);
                }
            } catch (cleanupError) {
                console.warn('Warning: Could not delete temp file:', cleanupError.message);
            }

            res.status(500).json({
                error: 'Failed to start emotion detection process',
                details: error.message
            });
        });

    } catch (error) {
        console.error('💥 Server error:', error);

        // Clean up temp file on error
        try {
            if (tempImagePath && fs.existsSync(tempImagePath)) {
                fs.unlinkSync(tempImagePath);
            }
        } catch (cleanupError) {
            console.warn('Warning: Could not delete temp file:', cleanupError.message);
        }

        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File size too large (max 5MB)' });
        }
    }
    console.error('🚫 Unhandled error:', error);
    res.status(500).json({ error: error.message });
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('🛑 Shutting down server...');

    // Clean up any remaining temp files
    const tempDir = path.join(__dirname, 'temp');
    if (fs.existsSync(tempDir)) {
        try {
            const files = fs.readdirSync(tempDir);
            files.forEach(file => {
                if (file.startsWith('temp_image_')) {
                    fs.unlinkSync(path.join(tempDir, file));
                }
            });
            console.log('🗑️ Cleaned up temp files');
        } catch (error) {
            console.warn('Warning: Error cleaning up temp files:', error.message);
        }
    }

    process.exit(0);
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Emotion Detection API with Age Recognition running on port ${PORT}`);
    console.log(`📍 Health check: http://localhost:${PORT}/health`);
    console.log(`📧 Emotion only: POST http://localhost:${PORT}/predict-emotion`);
    console.log(`🎵 With recommendations: POST http://localhost:${PORT}/predict-with-recommendations`);

    // Ensure temp directory exists
    const tempDir = path.join(__dirname, 'temp');
    if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir);
        console.log('📁 Created temp directory');
    }
});
