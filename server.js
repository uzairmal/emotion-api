const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// ---- FIX 6: Unified Python command ----
const PYTHON_CMD = process.env.PYTHON_CMD || 'python3';

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

        const tempDir = path.join(__dirname, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir);
        }

        const timestamp = Date.now();
        tempImagePath = path.join(tempDir, `temp_image_${timestamp}.jpg`);
        fs.writeFileSync(tempImagePath, req.file.buffer);
        console.log('💾 Image saved to temp file:', tempImagePath);

        const pythonProcess = spawn(PYTHON_CMD, ['predict_emotion.py', tempImagePath]);

        // ---- FIX 7: Timeout protection ----
        const PYTHON_TIMEOUT_MS = 30000;
        const timeoutId = setTimeout(() => {
            console.error('⏱️ Python process timeout — killing process');
            pythonProcess.kill('SIGKILL');
        }, PYTHON_TIMEOUT_MS);

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
            clearTimeout(timeoutId);

            try {
                if (tempImagePath && fs.existsSync(tempImagePath)) {
                    fs.unlinkSync(tempImagePath);
                    console.log('🗑️ Temp file cleaned up');
                }
            } catch (cleanupError) {
                console.warn('⚠️ Could not delete temp file:', cleanupError.message);
            }

            console.log(`🐍 Python process exited with code ${code}`);

            if (code !== 0) {
                return res.status(500).json({
                    error: 'Prediction failed',
                    details: errorData || 'Python process exited with non-zero code'
                });
            }

            if (!resultData.trim()) {
                return res.status(500).json({
                    error: 'No output from prediction model'
                });
            }

            try {
                const lines = resultData.trim().split('\n');
                const jsonLine = lines[lines.length - 1];
                const prediction = JSON.parse(jsonLine);

                if (prediction.status === 'error') {
                    return res.status(500).json(prediction);
                }

                res.json(prediction);
            } catch (parseError) {
                res.status(500).json({
                    error: 'Failed to parse prediction result',
                    details: parseError.message,
                    raw_output: resultData
                });
            }
        });

        pythonProcess.on('error', (error) => {
            clearTimeout(timeoutId);

            try {
                if (tempImagePath && fs.existsSync(tempImagePath)) {
                    fs.unlinkSync(tempImagePath);
                }
            } catch {}

            res.status(500).json({
                error: 'Failed to start prediction process',
                details: error.message
            });
        });

    } catch (error) {
        console.error('💥 Server error:', error);

        try {
            if (tempImagePath && fs.existsSync(tempImagePath)) {
                fs.unlinkSync(tempImagePath);
            }
        } catch {}

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
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const tempDir = path.join(__dirname, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir);
        }

        tempImagePath = path.join(tempDir, `temp_image_${Date.now()}.jpg`);
        fs.writeFileSync(tempImagePath, req.file.buffer);

        const pythonProcess = spawn(PYTHON_CMD, ['predict_emotion.py', tempImagePath]);

        // ---- FIX 7: Timeout protection ----
        const PYTHON_TIMEOUT_MS = 30000;
        const timeoutId = setTimeout(() => {
            console.error('⏱️ Python process timeout — killing process');
            pythonProcess.kill('SIGKILL');
        }, PYTHON_TIMEOUT_MS);

        let resultData = '';
        let errorData = '';

        pythonProcess.stdout.on('data', (data) => {
            resultData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorData += data.toString();
        });

        pythonProcess.on('close', (code) => {
            clearTimeout(timeoutId);

            try {
                if (tempImagePath && fs.existsSync(tempImagePath)) {
                    fs.unlinkSync(tempImagePath);
                }
            } catch {}

            if (code !== 0) {
                return res.status(500).json({
                    error: 'Emotion prediction failed',
                    details: errorData
                });
            }

            try {
                const lines = resultData.trim().split('\n');
                const jsonLine = lines[lines.length - 1];
                res.json(JSON.parse(jsonLine));
            } catch (err) {
                res.status(500).json({
                    error: 'Failed to parse prediction result',
                    raw_output: resultData
                });
            }
        });

        pythonProcess.on('error', (error) => {
            clearTimeout(timeoutId);
            res.status(500).json({
                error: 'Failed to start emotion detection process',
                details: error.message
            });
        });

    } catch (error) {
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError && error.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({ error: 'File size too large (max 5MB)' });
    }
    res.status(500).json({ error: error.message });
});

// Graceful shutdown
process.on('SIGINT', () => {
    const tempDir = path.join(__dirname, 'temp');
    if (fs.existsSync(tempDir)) {
        fs.readdirSync(tempDir).forEach(file => {
            if (file.startsWith('temp_image_')) {
                fs.unlinkSync(path.join(tempDir, file));
            }
        });
    }
    process.exit(0);
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});
