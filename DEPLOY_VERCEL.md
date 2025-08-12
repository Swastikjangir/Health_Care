# Deploying Frontend to Vercel

## Prerequisites
- Your backend is already deployed on Render at: https://health-care-aec6.onrender.com
- Your frontend code is in the `frontens` directory
- You have a Vercel account

## Method 1: Vercel Dashboard (Recommended)

### Step 1: Connect Repository
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository: `Swastikjangir/Health_Care`
4. Vercel will auto-detect it's a Vite/React project

### Step 2: Configure Project
1. **Framework Preset**: Vite (should be auto-detected)
2. **Root Directory**: `frontens`
3. **Build Command**: `npm run build`
4. **Output Directory**: `dist`
5. **Install Command**: `npm install`

### Step 3: Environment Variables
Add these environment variables:
```
VITE_API_BASE_URL=https://health-care-aec6.onrender.com
```

### Step 4: Deploy
1. Click "Deploy"
2. Wait for build to complete
3. Your app will be live at: `https://your-project-name.vercel.app`

## Method 2: Vercel CLI

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Navigate to Frontend Directory
```bash
cd frontens
```

### Step 3: Deploy
```bash
vercel
```

Follow the prompts:
- Set up and deploy? `Y`
- Which scope? `Select your account`
- Link to existing project? `N`
- Project name: `smart-health-frontend`
- In which directory is your code located? `./`
- Want to override the settings? `N`

## Method 3: GitHub Integration (Auto-Deploy)

### Step 1: Connect GitHub
1. In Vercel dashboard, go to "Settings" → "Git"
2. Connect your GitHub account
3. Select your repository

### Step 2: Configure Auto-Deploy
1. **Production Branch**: `main`
2. **Root Directory**: `frontens`
3. **Framework Preset**: Vite
4. **Build Command**: `npm run build`
5. **Output Directory**: `dist`

### Step 3: Environment Variables
Add the same environment variable:
```
VITE_API_BASE_URL=https://health-care-aec6.onrender.com
```

## Post-Deployment

### Test Your App
1. Visit your Vercel URL
2. Test the health prediction form
3. Verify API calls to Render backend
4. Check that all routes work (Home, Prediction, About)

### Custom Domain (Optional)
1. In Vercel dashboard, go to "Settings" → "Domains"
2. Add your custom domain
3. Configure DNS records as instructed

## Troubleshooting

### Build Errors
- Ensure you're in the `frontens` directory
- Check that all dependencies are in `package.json`
- Verify Node.js version compatibility

### API Connection Issues
- Check `VITE_API_BASE_URL` environment variable
- Verify backend is running on Render
- Check CORS settings in backend

### Routing Issues
- The `vercel.json` handles SPA routing
- All routes should redirect to `index.html`

## Success Indicators
✅ Frontend builds successfully  
✅ App loads without errors  
✅ API calls to Render backend work  
✅ All routes function properly  
✅ Health prediction form works  

Your full-stack app will be live with:
- **Frontend**: Vercel (fast, global CDN)
- **Backend**: Render (ML models, API)
- **Database**: In-memory (for demo purposes)
