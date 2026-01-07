const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 10000;

// --------------------
// PYTHON CONFIG
// --------------------
const PYTHON_CMD = process.env.PYTHON_CMD || 'python3';
const PYTHON_TIMEOUT_MS = 60000; // ✅ 60 seconds (IMPORTANT)

// --------------------
// MIDDLEWARE
// --------------------
app.use(cors());
app.use(express.json());

// --------------------
// MULTER CONFIG
// --------------------
const storage = multer.memoryStorage();
const upload = multer({
    storage,
    limits: { fileSize: 5 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) cb(null, true);
        else cb(new Error('Only image files allowed'));
    }
});

// --------------------
// HEALTH CHECK
// --------------------
app.get('/health', (req, res) => {
    res.json({
        status: 'OK',
        service: 'Emotion + Age + Music API',
        timestamp: new Date().toISOString()
    });
});

// --------------------
// MAIN ENDPOINT
// --------------------
app.post('/predict-with-recommendations', upload.single('image'), async (req, res) => {
    let tempImagePath;

    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image provided' });
        }

        const tempDir = path.join(__dirname, 'temp');
        if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir);

        tempImagePath = path.join(tempDir, `temp_image_${Date.now()}.jpg`);
        fs.writeFileSync(tempImagePath, req.file.buffer);

        console.log('💾 Image saved:', tempImagePath);

        // --------------------
        // STEP 3 LOGIC ADDED HERE
        // --------------------
        const pythonProcess = spawn(
            PYTHON_CMD,
            ['predict_emotion.py', tempImagePath],
            {
                env: {
                    ...process.env,
                    SKIP_AGE: '1' // ✅ STEP-3: disable age detection
                }
            }
        );

        const timeoutId = setTimeout(() => {
            console.error('⏱️ Python timeout — killed');
            pythonProcess.kill('SIGKILL');
        }, PYTHON_TIMEOUT_MS);

        let stdout = '';
        let stderr = '';

        pythonProcess.stdout.on('data', data => {
            stdout += data.toString();
        });

        pythonProcess.stderr.on('data', data => {
            stderr += data.toString();
            console.error('🐍 stderr:', data.toString());
        });

        pythonProcess.on('close', code => {
            clearTimeout(timeoutId);

            if (fs.existsSync(tempImagePath)) fs.unlinkSync(tempImagePath);

            if (code !== 0) {
                return res.status(500).json({
                    error: 'Python process failed',
                    details: stderr
                });
            }

            try {
                const lines = stdout.trim().split('\n');
                const json = JSON.parse(lines[lines.length - 1]);
                return res.json(json);
            } catch (err) {
                return res.status(500).json({
                    error: 'Invalid Python output',
                    raw: stdout
                });
            }
        });

        pythonProcess.on('error', err => {
            clearTimeout(timeoutId);
            if (fs.existsSync(tempImagePath)) fs.unlinkSync(tempImagePath);

            res.status(500).json({
                error: 'Failed to start Python',
                details: err.message
            });
        });

    } catch (err) {
        if (tempImagePath && fs.existsSync(tempImagePath)) fs.unlinkSync(tempImagePath);
        res.status(500).json({ error: err.message });
    }
});

// --------------------
// SERVER START
// --------------------
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});
