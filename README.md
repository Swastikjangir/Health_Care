# Smart Health Risk Prediction & Recommendation System

A comprehensive AI-powered health risk assessment system that combines machine learning, data analysis, and personalized recommendations to help users understand their health risks and take preventive actions.

## 🚀 Features

- **AI-Powered Health Risk Assessment**: Multiple ML models for disease prediction
- **Patient Clustering**: Group patients by health similarity using advanced clustering algorithms
- **Association Rule Mining**: Discover hidden health pattern relationships
- **Personalized Recommendations**: Tailored diet, exercise, and lifestyle advice
- **Comprehensive Reports**: Detailed health insights with visualizations
- **Privacy-First Design**: No data storage, end-to-end encryption

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **Data Preprocessing**: Advanced data cleaning, encoding, and scaling
- **Machine Learning Models**: Classification, regression, clustering, and association rules
- **Recommendation Engine**: Personalized health recommendations
- **API Layer**: RESTful API with FastAPI framework

### Frontend (React.js)
- **Modern UI**: Built with React 19 and Tailwind CSS
- **Interactive Forms**: Multi-step health data input
- **Data Visualization**: Charts and graphs for health insights
- **Responsive Design**: Mobile-first approach

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **FastAPI** - Modern, fast web framework
- **Scikit-learn** - Machine learning library
- **Pandas & NumPy** - Data manipulation
- **XGBoost** - Gradient boosting framework
- **TensorFlow/Keras** - Deep learning (optional)
- **Plotly** - Interactive visualizations

### Frontend
- **React.js 19** - Modern React with hooks
- **Tailwind CSS** - Utility-first CSS framework
- **Chart.js** - Charting library
- **React Hook Form** - Form handling
- **Lucide React** - Icon library

### Deployment
- **Backend**: Heroku, Render, or Docker
- **Frontend**: Vercel, Netlify, or static hosting
- **Streamlit**: Alternative ML demo interface

## 📁 Project Structure

```
SmartHealthRiskPrediction/
├── backend/
│   ├── data/
│   │   ├── raw/                      # Original datasets
│   │   ├── processed/                # Cleaned datasets
│   │   └── models/                   # Saved trained models
│   ├── notebooks/                    # Jupyter notebooks
│   ├── src/
│   │   ├── preprocessing.py          # Data preprocessing
│   │   ├── classification_models.py  # Classification algorithms
│   │   ├── regression_models.py      # Regression algorithms
│   │   ├── clustering.py             # Clustering algorithms
│   │   ├── association_rules.py      # Association rule mining
│   │   ├── recommend.py              # Recommendation engine
│   │   └── utils.py                  # Utility functions
│   ├── app.py                        # FastAPI main application
│   └── requirements.txt              # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/               # React components
│   │   ├── pages/                    # Page components
│   │   ├── context/                  # React context
│   │   └── App.jsx                   # Main app component
│   ├── package.json                  # Node.js dependencies
│   └── tailwind.config.js            # Tailwind configuration
└── README.md
```

## 🚀 Quick Start

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server**:
   ```bash
   python app.py
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:5173`

## 📊 API Endpoints

### Health Assessment
- `POST /predict/health-risk` - Predict health risks and get recommendations
- `POST /cluster/patients` - Cluster patients by health parameters
- `GET /models/available` - Get available ML models
- `GET /features/health-parameters` - Get supported health parameters

### Data Management
- `POST /upload/dataset` - Upload datasets for training
- `GET /recommendations/templates` - Get recommendation templates

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the backend directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_PATH=./data/models
DEFAULT_MODEL=random_forest

# Security
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
```

### Model Parameters
Adjust ML model parameters in the respective model files:
- `classification_models.py` - Classification algorithm settings
- `clustering.py` - Clustering algorithm parameters
- `regression_models.py` - Regression model configurations

## 📈 Machine Learning Models

### Classification Models
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble tree-based classification
- **XGBoost**: Gradient boosting for high performance
- **Support Vector Machine**: Kernel-based classification
- **Neural Networks**: Deep learning classification

### Clustering Models
- **K-Means**: Partition-based clustering
- **Hierarchical Clustering**: Tree-based clustering
- **DBSCAN**: Density-based clustering
- **Gaussian Mixture**: Probabilistic clustering

### Regression Models
- **Linear Regression**: Basic linear modeling
- **Ridge/Lasso**: Regularized regression
- **Random Forest Regressor**: Tree-based regression
- **XGBoost Regressor**: Gradient boosting regression

## 🎯 Usage Examples

### Health Risk Assessment
```python
import requests

# Patient data
patient_data = {
    "age": 45,
    "gender": "male",
    "blood_pressure": 140,
    "glucose": 120,
    "bmi": 28,
    "smoking": "former_smoker"
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict/health-risk",
    json={"patient_data": patient_data}
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Risk Score: {result['risk_score']}%")
```

### Patient Clustering
```python
# Multiple patients
patients = [
    {"age": 45, "blood_pressure": 140, "glucose": 120},
    {"age": 52, "blood_pressure": 135, "glucose": 110},
    {"age": 38, "blood_pressure": 120, "glucose": 95}
]

response = requests.post(
    "http://localhost:8000/cluster/patients",
    json=patients
)

clusters = response.json()
print(f"Number of clusters: {clusters['n_clusters']}")
```

## 🔒 Security & Privacy

- **No Data Storage**: Patient data is processed in memory only
- **End-to-End Encryption**: All data transmission is encrypted
- **HIPAA Compliance**: Built following healthcare privacy standards
- **Input Validation**: Comprehensive data validation and sanitization
- **CORS Protection**: Configurable cross-origin resource sharing

## 📊 Data Sources

The system is designed to work with various health datasets:
- **UCI Machine Learning Repository**: Diabetes, heart disease datasets
- **Kaggle**: Medical datasets and competitions
- **Custom Data**: Hospital records, clinical trials data

## 🚀 Deployment

### Backend Deployment
1. **Heroku**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

2. **Docker**:
   ```bash
   docker build -t smart-health-backend .
   docker run -p 8000:8000 smart-health-backend
   ```

3. **Render**: Connect GitHub repository and deploy

### Frontend Deployment
1. **Vercel**:
   ```bash
   npm install -g vercel
   vercel --prod
   ```

2. **Netlify**: Drag and drop build folder
3. **Static Hosting**: Upload build files to any web server

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: This system is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 📞 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Check the code comments and docstrings
- **Community**: Join our discussions and share your experiences

## 🔮 Future Enhancements

- **Deep Learning**: CNN models for medical image analysis
- **Real-time Monitoring**: Continuous health data streaming
- **Mobile App**: React Native mobile application
- **Integration**: EHR system integrations
- **Advanced Analytics**: Predictive analytics dashboard

---

**Built with ❤️ for better health outcomes through AI**
