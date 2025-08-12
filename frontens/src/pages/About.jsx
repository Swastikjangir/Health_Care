import React from 'react';
import { 
  Brain, 
  Shield, 
  Users, 
  Code, 
  Database, 
  Lock,
  TrendingUp,
  Heart,
  Zap,
  Target
} from 'lucide-react';

const About = () => {
  const features = [
    {
      icon: Brain,
      title: 'Machine Learning Models',
      description: 'Advanced algorithms including Random Forest, XGBoost, Neural Networks, and more for accurate health predictions.',
      color: 'text-blue-600'
    },
    {
      icon: Database,
      title: 'Data Processing',
      description: 'Sophisticated preprocessing pipeline with feature engineering, outlier detection, and data validation.',
      color: 'text-green-600'
    },
    {
      icon: Users,
      title: 'Patient Clustering',
      description: 'K-means and hierarchical clustering to identify patient groups with similar health profiles.',
      color: 'text-purple-600'
    },
    {
      icon: TrendingUp,
      title: 'Association Rules',
      description: 'Apriori algorithm to discover hidden health pattern relationships and co-occurring conditions.',
      color: 'text-orange-600'
    },
    {
      icon: Shield,
      title: 'Privacy First',
      description: 'End-to-end encryption, no data storage, and HIPAA-compliant security measures.',
      color: 'text-red-600'
    },
    {
      icon: Target,
      title: 'Personalized Insights',
      description: 'AI-generated recommendations tailored to individual health profiles and risk factors.',
      color: 'text-indigo-600'
    }
  ];

  const techStack = [
    {
      category: 'Backend & ML',
      technologies: ['Python', 'Scikit-learn', 'Pandas', 'NumPy', 'XGBoost', 'TensorFlow/Keras']
    },
    {
      category: 'API & Framework',
      technologies: ['FastAPI', 'Uvicorn', 'Pydantic', 'CORS Middleware']
    },
    {
      category: 'Frontend',
      technologies: ['React.js', 'Tailwind CSS', 'Chart.js', 'React Hook Form']
    },
    {
      category: 'Data Visualization',
      technologies: ['Plotly', 'Matplotlib', 'Seaborn', 'Recharts']
    },
    {
      category: 'Deployment',
      technologies: ['Docker', 'Heroku/Render', 'Vercel', 'Streamlit']
    }
  ];

  const mlModels = [
    {
      name: 'Classification Models',
      description: 'Predict disease risk and health outcomes',
      models: ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM', 'Neural Networks']
    },
    {
      name: 'Regression Models',
      description: 'Predict continuous health values',
      models: ['Linear Regression', 'Ridge/Lasso', 'Random Forest Regressor', 'XGBoost Regressor']
    },
    {
      name: 'Clustering Models',
      description: 'Group patients by health similarity',
      models: ['K-Means', 'Hierarchical Clustering', 'DBSCAN', 'Gaussian Mixture']
    },
    {
      name: 'Association Rules',
      description: 'Discover health pattern relationships',
      models: ['Apriori Algorithm', 'FP-Growth', 'Market Basket Analysis']
    }
  ];

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center py-20 bg-gradient-to-br from-primary-50 to-blue-50 rounded-3xl">
        <div className="max-w-4xl mx-auto px-4">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
            About{' '}
            <span className="text-gradient">Smart Health</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            A cutting-edge health risk prediction system powered by artificial intelligence and machine learning, 
            designed to provide personalized health insights and preventive recommendations.
          </p>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="grid md:grid-cols-2 gap-8">
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <Target className="h-8 w-8 text-primary-600" />
            <h3 className="text-2xl font-bold text-gray-900">Our Mission</h3>
          </div>
          <p className="text-gray-600 leading-relaxed">
            To democratize access to advanced health risk assessment by leveraging the power of artificial intelligence, 
            enabling individuals to take proactive control of their health through early detection and personalized prevention strategies.
          </p>
        </div>

        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <Zap className="h-8 w-8 text-primary-600" />
            <h3 className="text-2xl font-bold text-gray-900">Our Vision</h3>
          </div>
          <p className="text-gray-600 leading-relaxed">
            A world where AI-powered health insights are accessible to everyone, leading to earlier disease detection, 
            better preventive care, and improved overall population health outcomes.
          </p>
        </div>
      </section>

      {/* Key Features */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Advanced AI Capabilities
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Our system combines state-of-the-art machine learning with medical expertise 
            to deliver comprehensive health insights.
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div key={index} className="card hover:shadow-lg transition-all duration-300 group">
                <div className={`mb-4 ${feature.color}`}>
                  <Icon className="h-12 w-12 group-hover:scale-110 transition-transform duration-300" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Technology Stack */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Technology Stack
          </h2>
          <p className="text-lg text-gray-600">
            Built with modern, scalable technologies for reliable performance
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {techStack.map((tech, index) => (
            <div key={index} className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                {tech.category}
              </h3>
              <div className="space-y-2">
                {tech.technologies.map((technology, techIndex) => (
                  <div key={techIndex} className="flex items-center space-x-2">
                    <Code className="h-4 w-4 text-primary-600" />
                    <span className="text-gray-700">{technology}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ML Models */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Machine Learning Models
          </h2>
          <p className="text-lg text-gray-600">
            Comprehensive suite of AI models for different health prediction tasks
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8">
          {mlModels.map((model, index) => (
            <div key={index} className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {model.name}
              </h3>
              <p className="text-gray-600 mb-4">
                {model.description}
              </p>
              <div className="space-y-2">
                {model.models.map((mlModel, modelIndex) => (
                  <div key={modelIndex} className="flex items-center space-x-2">
                    <Brain className="h-4 w-4 text-primary-600" />
                    <span className="text-gray-700">{mlModel}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Privacy & Security */}
      <section className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-3xl p-12 text-white">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <Lock className="h-16 w-16 text-white" />
          </div>
          <h2 className="text-3xl font-bold mb-4">
            Privacy & Security First
          </h2>
          <p className="text-xl opacity-90">
            Your health data security is our top priority
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <Shield className="h-12 w-12 mx-auto mb-3 opacity-80" />
            <h3 className="text-lg font-semibold mb-2">End-to-End Encryption</h3>
            <p className="text-sm opacity-80">
              All data is encrypted during transmission and processing
            </p>
          </div>
          
          <div className="text-center">
            <Database className="h-12 w-12 mx-auto mb-3 opacity-80" />
            <h3 className="text-lg font-semibold mb-2">No Data Storage</h3>
            <p className="text-sm opacity-80">
              Your personal information is never stored on our servers
            </p>
          </div>
          
          <div className="text-center">
            <Users className="h-12 w-12 mx-auto mb-3 opacity-80" />
            <h3 className="text-lg font-semibold mb-2">HIPAA Compliant</h3>
            <p className="text-sm opacity-80">
              Built following healthcare privacy standards and regulations
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            How the System Works
          </h2>
          <p className="text-lg text-gray-600">
            From data input to personalized insights
          </p>
        </div>
        
        <div className="grid md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-primary-600">1</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Data Input</h3>
            <p className="text-sm text-gray-600">
              Users provide health parameters through our secure form interface
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-primary-600">2</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">AI Processing</h3>
            <p className="text-sm text-gray-600">
              Multiple ML models analyze the data for patterns and risk factors
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-primary-600">3</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Risk Assessment</h3>
            <p className="text-sm text-gray-600">
              Comprehensive health risk evaluation with confidence scores
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-primary-600">4</span>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Personalized Output</h3>
            <p className="text-sm text-gray-600">
              Actionable recommendations and detailed health reports
            </p>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="text-center p-8 bg-yellow-50 border border-yellow-200 rounded-xl">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <Heart className="h-6 w-6 text-yellow-600" />
          <span className="text-lg font-semibold text-yellow-800">Important Notice</span>
        </div>
        <p className="text-yellow-700 max-w-3xl mx-auto">
          Smart Health is an educational tool designed to provide general health insights and recommendations. 
          It is not a substitute for professional medical advice, diagnosis, or treatment. 
          Always consult with qualified healthcare professionals for medical concerns and before making 
          any changes to your health routine.
        </p>
      </section>
    </div>
  );
};

export default About;
