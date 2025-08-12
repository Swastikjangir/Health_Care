# Deploying Backend to Render

## Quick Fix for Build Issues

The main issue is Python 3.13 compatibility. Use these settings in Render:

### 1. Environment Variables
```
PYTHON_VERSION=3.11.9
```

### 2. Build Command
```bash
pip install -r requirements-render.txt
```

### 3. Start Command
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### 4. Render Settings
- **Runtime**: Python 3.11
- **Build Command**: `pip install -r requirements-render.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  ```
  FRONTEND_ORIGINS=https://your-frontend.vercel.app
  ENABLE_HEAVY=true
  ```

### 5. Alternative: Use requirements.txt
If you want to use the main requirements.txt, change the build command to:
```bash
pip install -r requirements.txt
```

## Why This Happens
- Python 3.13 is very new and many ML packages haven't been updated yet
- pandas 2.1.3 has C compilation issues with Python 3.13
- TensorFlow and other packages may have similar compatibility issues

## Recommended Approach
1. Use `requirements-render.txt` for guaranteed compatibility
2. Set Python version to 3.11 in Render
3. This gives you all features while ensuring successful builds

## If You Still Get Build Errors
Consider using the "light mode" approach:
1. Set `ENABLE_HEAVY=false` in environment variables
2. Use `requirements-light.txt` (I can create this if needed)
3. This disables heavy ML features but keeps the core API working
