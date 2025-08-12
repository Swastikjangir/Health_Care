import React, { useState } from 'react';
import { useHealth } from '../context/HealthContext';
import HealthForm from '../components/HealthForm';
import RiskResult from '../components/RiskResult';
import { Activity, Loader2, AlertCircle } from 'lucide-react';

const Prediction = () => {
  const { loading, error, healthAssessment, api, dispatch } = useHealth();
  const [showResults, setShowResults] = useState(false);

  const handleFormSubmit = async (formData) => {
    try {
      await api.predictHealthRisk(formData);
      setShowResults(true);
    } catch (error) {
      console.error('Error submitting form:', error);
    }
  };

  const handleNewAssessment = () => {
    setShowResults(false);
    dispatch({ type: 'CLEAR_DATA' });
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Health Risk Assessment
        </h1>
        <p className="text-lg text-gray-600">
          Enter your health parameters to receive AI-powered risk analysis and personalized recommendations
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <span className="text-red-700 font-medium">Error: {error}</span>
          </div>
        </div>
      )}

      {loading && (
        <div className="text-center py-12">
          <Loader2 className="h-12 w-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-lg text-gray-600">Analyzing your health data...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
        </div>
      )}

      {!loading && !showResults && !healthAssessment && (
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Form Section */}
          <div className="lg:col-span-2">
            <HealthForm onSubmit={handleFormSubmit} />
          </div>

          {/* Info Section */}
          <div className="space-y-6">
            <div className="card">
              <div className="flex items-center space-x-3 mb-4">
                <Activity className="h-6 w-6 text-primary-600" />
                <h3 className="text-lg font-semibold text-gray-900">How It Works</h3>
              </div>
              <div className="space-y-3 text-sm text-gray-600">
                <p>1. Enter your health parameters in the form</p>
                <p>2. Our AI models analyze your data</p>
                <p>3. Get personalized risk assessment</p>
                <p>4. Receive actionable recommendations</p>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">What We Analyze</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• Age, gender, and basic demographics</p>
                <p>• Blood pressure and heart health</p>
                <p>• Glucose levels and diabetes risk</p>
                <p>• BMI and weight-related factors</p>
                <p>• Lifestyle factors (smoking, alcohol, exercise)</p>
                <p>• Stress levels and sleep patterns</p>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Privacy & Security</h3>
              <div className="space-y-2 text-sm text-gray-600">
                <p>• Your data is encrypted and secure</p>
                <p>• No personal information is stored</p>
                <p>• Results are for educational purposes only</p>
                <p>• Always consult healthcare professionals</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {showResults && healthAssessment && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold text-gray-900">Your Health Assessment Results</h2>
            <button
              onClick={handleNewAssessment}
              className="btn-secondary"
            >
              New Assessment
            </button>
          </div>
          
          <RiskResult assessment={healthAssessment} />
        </div>
      )}
    </div>
  );
};

export default Prediction;
