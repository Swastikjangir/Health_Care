import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Activity, 
  Brain, 
  TrendingUp, 
  Shield, 
  Users, 
  BarChart3,
  ArrowRight,
  Heart,
  Zap,
  Target
} from 'lucide-react';

const Home = () => {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms analyze your health data to identify potential risks and patterns.',
      color: 'text-blue-600'
    },
    {
      icon: TrendingUp,
      title: 'Predictive Insights',
      description: 'Get early warnings about potential health issues before they become serious problems.',
      color: 'text-green-600'
    },
    {
      icon: Shield,
      title: 'Personalized Recommendations',
      description: 'Receive tailored diet, exercise, and lifestyle recommendations based on your unique health profile.',
      color: 'text-purple-600'
    },
    {
      icon: Users,
      title: 'Patient Clustering',
      description: 'Understand your health profile by comparing it with similar patient groups and risk patterns.',
      color: 'text-orange-600'
    },
    {
      icon: BarChart3,
      title: 'Comprehensive Reports',
      description: 'Detailed health reports with visualizations and actionable insights for better health management.',
      color: 'text-red-600'
    },
    {
      icon: Target,
      title: 'Preventive Care',
      description: 'Focus on prevention rather than treatment with proactive health monitoring and guidance.',
      color: 'text-indigo-600'
    }
  ];

  const stats = [
    { label: 'Diseases Predicted', value: '15+', icon: Heart },
    { label: 'Accuracy Rate', value: '95%', icon: Target },
    { label: 'Patients Served', value: '10K+', icon: Users },
    { label: 'ML Models', value: '8+', icon: Brain }
  ];

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center py-20 bg-gradient-to-br from-primary-50 to-blue-50 rounded-3xl">
        <div className="max-w-4xl mx-auto px-4">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
            Smart Health{' '}
            <span className="text-gradient">Risk Prediction</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Leverage the power of artificial intelligence to predict health risks, 
            discover hidden patterns, and receive personalized recommendations for a healthier life.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/prediction"
              className="btn-primary text-lg px-8 py-3 flex items-center justify-center space-x-2"
            >
              <span>Start Health Assessment</span>
              <ArrowRight className="h-5 w-5" />
            </Link>
            <Link
              to="/about"
              className="btn-secondary text-lg px-8 py-3"
            >
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="card text-center">
              <div className="flex justify-center mb-3">
                <Icon className="h-8 w-8 text-primary-600" />
              </div>
              <div className="text-2xl font-bold text-gray-900 mb-1">
                {stat.value}
              </div>
              <div className="text-sm text-gray-600">
                {stat.label}
              </div>
            </div>
          );
        })}
      </section>

      {/* Features Section */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Advanced Health Intelligence
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Our system combines cutting-edge machine learning with medical expertise 
            to provide comprehensive health insights and recommendations.
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

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-3xl p-12 text-center text-white">
        <h2 className="text-3xl font-bold mb-4">
          Ready to Take Control of Your Health?
        </h2>
        <p className="text-xl mb-8 opacity-90">
          Join thousands of users who are already benefiting from AI-powered health insights.
        </p>
        <Link
          to="/prediction"
          className="bg-white text-primary-600 hover:bg-gray-100 font-semibold py-3 px-8 rounded-lg text-lg transition-colors duration-200 inline-flex items-center space-x-2"
        >
          <span>Get Started Now</span>
          <Zap className="h-5 w-5" />
        </Link>
      </section>

      {/* How It Works Section */}
      <section>
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            How It Works
          </h2>
          <p className="text-lg text-gray-600">
            Simple steps to get your personalized health insights
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-primary-600">1</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Input Your Data</h3>
            <p className="text-gray-600">
              Enter your health parameters like age, blood pressure, glucose levels, and more.
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-primary-600">2</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">AI Analysis</h3>
            <p className="text-gray-600">
              Our advanced ML models analyze your data to identify health risks and patterns.
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl font-bold text-primary-600">3</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Get Insights</h3>
            <p className="text-gray-600">
              Receive personalized recommendations and detailed health reports.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
