# Deploying Backend to Render

## IMPORTANT: Fixed Multiple Dependency Conflicts

I've resolved two major dependency conflicts:
1. **TensorFlow vs FastAPI/Pydantic** - Fixed with compatible versions
2. **Pillow version conflicts** - Fixed with Pillow 9.5.0

### What Was Fixed
- **TensorFlow 2.13.0** requires `typing-extensions<4.6.0`
- **FastAPI 0.104.1** and **Pydantic 2.0.3** need `typing-extensions>=4.6.1`
- **Solution**: Use FastAPI 0.95.2 + Pydantic 1.10.8

- **Pillow conflicts**: matplotlib, reportlab, and streamlit had conflicting Pillow requirements
- **Solution**: Use Pillow 9.5.0 (compatible with all packages)

### Method 1: Use render.yaml (Recommended)
1. **Commit and push** the updated `render.yaml` file to your repository
2. **Connect your GitHub repo** to Render
3. **Render will automatically detect** the `render.yaml` and use Python 3.11

### Method 2: Manual Render Dashboard Settings
If you prefer manual setup:

1. **Environment Variables** (set these FIRST):
   ```
   PYTHON_VERSION=3.11.9
   FRONTEND_ORIGINS=https://health-care-roan-nu.vercel.app
   ENABLE_HEAVY=true
   ```

2. **Build Command**:
   ```bash
   pip install -r requirements-render.txt
   ```

3. **Start Command**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

4. **Python Version**: Select Python 3.11 from the dropdown

### Method 3: Minimal Deployment (If Still Having Issues)
If you continue to have dependency conflicts:

1. **Build Command**: `pip install -r requirements-minimal.txt`
2. **This removes**: TensorFlow, XGBoost, OpenCV, matplotlib, seaborn
3. **Keeps**: Core API, recommendations, basic ML (scikit-learn only)
4. **Set**: `ENABLE_HEAVY=false` in environment variables

### Method 4: Ultra-Minimal Deployment (Last Resort)
If all else fails:

1. **Build Command**: `pip install -r requirements-ultra-minimal.txt`
2. **This removes**: All ML packages, keeps only core API
3. **Keeps**: FastAPI, recommendations engine, basic functionality
4. **Set**: `ENABLE_HEAVY=false` in environment variables

## Why This Happens
- **Python 3.13 is too new** for most ML packages
- **Package dependency conflicts** between different versions
- **TensorFlow has strict version requirements** for dependencies
- **Multiple packages can have conflicting requirements** for the same dependency
- **render.yaml is the most reliable** way to specify Python version

## Current Status
✅ `requirements-render.txt` - Fixed all dependency conflicts  
✅ `requirements-minimal.txt` - Backup with basic ML  
✅ `requirements-ultra-minimal.txt` - Core API only  
✅ `render.yaml` - Forces Python 3.11 + backend directory + Vercel domain  
✅ `runtime.txt` - Backup Python specification  
✅ `.python-version` - Additional Python version hint  

## Next Steps
1. **Commit all updated files** to your repository
2. **Push to GitHub**
3. **Connect to Render** (it will auto-detect render.yaml)
4. **Deploy** - should work now!

## If You Still Get Errors
Try the requirements files in this order:
1. `requirements-render.txt` (full features)
2. `requirements-minimal.txt` (basic ML)
3. `requirements-ultra-minimal.txt` (core API only)

Each removes more packages to eliminate dependency conflicts while maintaining functionality.

## Frontend Integration
Your frontend is now deployed at: **https://health-care-roan-nu.vercel.app/**
The backend CORS is configured to allow requests from this domain.
