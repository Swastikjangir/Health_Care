# Deploying Backend to Render

## IMPORTANT: Force Python 3.11

The main issue is that Render keeps using Python 3.13 despite our settings. Here's how to fix it:

### Method 1: Use render.yaml (Recommended)
1. **Commit and push** the `render.yaml` file to your repository
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

### Method 3: Use requirements.txt with Python 3.11
If you want to use the main requirements.txt:

1. **Set Python version to 3.11** in Render dashboard
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Why This Happens
- **Python 3.13 is too new** for most ML packages
- **TensorFlow 2.15+ doesn't support Python 3.13** yet
- **Render sometimes ignores runtime.txt** files
- **render.yaml is the most reliable** way to specify Python version

## Current Status
✅ `requirements-render.txt` - Python 3.11 compatible packages  
✅ `render.yaml` - Forces Python 3.11  
✅ `runtime.txt` - Backup Python specification  
✅ `.python-version` - Additional Python version hint  

## Next Steps
1. **Commit all files** to your repository
2. **Push to GitHub**
3. **Connect to Render** (it will auto-detect render.yaml)
4. **Deploy** - should work now!

## If You Still Get Errors
The `render.yaml` approach should work. If not, we can create a minimal `requirements-light.txt` that only includes essential packages.
