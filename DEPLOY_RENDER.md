# Deploying Backend to Render

## IMPORTANT: Fixed Dependency Conflicts

The main issue was a dependency conflict between TensorFlow 2.13.0 and newer FastAPI/Pydantic versions. I've fixed this by using compatible package versions.

### What Was Fixed
- **TensorFlow 2.13.0** requires `typing-extensions<4.6.0`
- **FastAPI 0.104.1** and **Pydantic 2.0.3** need `typing-extensions>=4.6.1`
- **Solution**: Use FastAPI 0.95.2 + Pydantic 1.10.8 (compatible with TensorFlow 2.13.0)

### Method 1: Use render.yaml (Recommended)
1. **Commit and push** the updated `render.yaml` file to your repository
2. **Connect your GitHub repo** to Render
3. **Render will automatically detect** the `render.yaml` and use Python 3.11

### Method 2: Manual Render Dashboard Settings
If you prefer manual setup:

1. **Environment Variables** (set these FIRST):
   ```
   PYTHON_VERSION=3.11.9
   FRONTEND_ORIGINS=https://your-frontend.vercel.app
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

## Why This Happens
- **Python 3.13 is too new** for most ML packages
- **Package dependency conflicts** between different versions
- **TensorFlow has strict version requirements** for dependencies
- **render.yaml is the most reliable** way to specify Python version

## Current Status
✅ `requirements-render.txt` - Fixed dependency conflicts  
✅ `requirements-minimal.txt` - Backup minimal packages  
✅ `render.yaml` - Forces Python 3.11 + backend directory  
✅ `runtime.txt` - Backup Python specification  
✅ `.python-version` - Additional Python version hint  

## Next Steps
1. **Commit all updated files** to your repository
2. **Push to GitHub**
3. **Connect to Render** (it will auto-detect render.yaml)
4. **Deploy** - should work now!

## If You Still Get Errors
The updated `requirements-render.txt` should resolve the dependency conflicts. If not, use `requirements-minimal.txt` for a lighter deployment that still provides core functionality.
