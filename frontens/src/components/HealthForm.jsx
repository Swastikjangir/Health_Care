import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { Heart, User, Activity, Shield, Coffee, Moon, Zap } from 'lucide-react';

const HealthForm = ({ onSubmit }) => {
  const { register, handleSubmit, formState: { errors }, watch } = useForm();
  const [currentStep, setCurrentStep] = useState(1);
  const totalSteps = 4;

  // Watch height and weight for BMI calculation
  const height = watch('height');
  const weight = watch('weight');

  // Calculate BMI when both height and weight are available
  const calculateBMI = () => {
    if (height && weight && height > 0 && weight > 0) {
      const heightInMeters = height / 100;
      return (weight / (heightInMeters * heightInMeters)).toFixed(1);
    }
    return null;
  };

  const formSections = [
    {
      step: 1,
      title: 'Basic Information',
      icon: User,
      fields: [
        { name: 'age', label: 'Age', type: 'number', required: true, min: 1, max: 120 },
        { name: 'gender', label: 'Gender', type: 'select', required: true, options: ['male', 'female', 'other'] },
        { name: 'height', label: 'Height (cm)', type: 'number', required: true, min: 100, max: 250 },
        { name: 'weight', label: 'Weight (kg)', type: 'number', required: true, min: 20, max: 300 }
      ]
    },
    {
      step: 2,
      title: 'Vital Signs',
      icon: Heart,
      fields: [
        { name: 'blood_pressure', label: 'Blood Pressure (mmHg)', type: 'number', required: true, min: 50, max: 300 },
        { name: 'glucose', label: 'Glucose Level (mg/dL)', type: 'number', required: true, min: 20, max: 1000 },
        { name: 'cholesterol', label: 'Cholesterol (mg/dL)', type: 'number', min: 50, max: 1000 },
        { name: 'creatinine', label: 'Creatinine (mg/dL)', type: 'number', min: 0.1, max: 10, step: 0.1 }
      ]
    },
    {
      step: 3,
      title: 'Lifestyle Factors',
      icon: Activity,
      fields: [
        { name: 'smoking', label: 'Smoking Status', type: 'select', required: true, options: ['never_smoker', 'former_smoker', 'current_smoker'] },
        { name: 'alcohol', label: 'Alcohol Consumption', type: 'select', required: true, options: ['minimal', 'moderate', 'excessive'] },
        { name: 'exercise_frequency', label: 'Exercise Frequency', type: 'select', required: true, options: ['daily', 'weekly', 'monthly', 'rarely'] },
        { name: 'sleep_hours', label: 'Sleep Hours per Night', type: 'number', required: true, min: 3, max: 12, step: 0.5 }
      ]
    },
    {
      step: 4,
      title: 'Additional Factors',
      icon: Shield,
      fields: [
        { name: 'stress_level', label: 'Stress Level', type: 'select', required: true, options: ['low', 'medium', 'high'] }
      ]
    }
  ];

  const currentSection = formSections.find(section => section.step === currentStep);

  const nextStep = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const onFormSubmit = (data) => {
    // Filter out empty values
    const filteredData = Object.fromEntries(
      Object.entries(data).filter(([_, value]) => value !== '' && value !== null)
    );
    
    // Calculate and add BMI
    const bmi = calculateBMI();
    if (bmi) {
      filteredData.bmi = parseFloat(bmi);
    }
    
    // Ensure all required fields are present with defaults if missing
    const processedData = {
      age: filteredData.age || 30,
      gender: filteredData.gender || 'male',
      blood_pressure: filteredData.blood_pressure || 120,
      glucose: filteredData.glucose || 100,
      bmi: filteredData.bmi || 25,
      cholesterol: filteredData.cholesterol || 180,
      creatinine: filteredData.creatinine || 1.0,
      smoking: filteredData.smoking || 'never_smoker',
      alcohol: filteredData.alcohol || 'minimal',
      sleep_hours: filteredData.sleep_hours || 7,
      stress_level: filteredData.stress_level || 'low',
      exercise_frequency: filteredData.exercise_frequency || 'weekly'
    };
    
    onSubmit(processedData);
  };

  const renderField = (field) => {
    const { name, label, type, required, options, min, max, step } = field;
    
    if (type === 'select') {
      return (
        <select
          {...register(name, { required: required && `${label} is required` })}
          className="input-field"
        >
          <option value="">Select {label}</option>
          {options.map(option => (
            <option key={option} value={option}>
              {option.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </option>
          ))}
        </select>
      );
    }

    return (
      <input
        type={type}
        {...register(name, { 
          required: required && `${label} is required`,
          min: min && { value: min, message: `${label} must be at least ${min}` },
          max: max && { value: max, message: `${label} must be at most ${max}` }
        })}
        min={min}
        max={max}
        step={step}
        placeholder={`Enter ${label.toLowerCase()}`}
        className="input-field"
      />
    );
  };

  return (
    <div className="card">
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          {formSections.map((section, index) => {
            const Icon = section.icon;
            return (
              <div key={section.step} className="flex flex-col items-center">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-2 ${
                  currentStep >= section.step 
                    ? 'bg-primary-600 text-white' 
                    : 'bg-gray-200 text-gray-500'
                }`}>
                  <Icon className="h-6 w-6" />
                </div>
                <span className={`text-sm font-medium ${
                  currentStep >= section.step ? 'text-primary-600' : 'text-gray-500'
                }`}>
                  {section.title}
                </span>
              </div>
            );
          })}
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-primary-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${(currentStep / totalSteps) * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Form Content */}
      <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-6">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            {currentSection.title}
          </h2>
          <p className="text-gray-600">
            Step {currentStep} of {totalSteps}
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {currentSection.fields.map((field) => (
            <div key={field.name} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                {field.label}
                {field.required && <span className="text-red-500 ml-1">*</span>}
              </label>
              {renderField(field)}
              {errors[field.name] && (
                <p className="text-sm text-red-600">{errors[field.name].message}</p>
              )}
            </div>
          ))}
        </div>

        {/* BMI Display */}
        {height && weight && calculateBMI() && (
          <div className="p-4 bg-green-50 rounded-lg border border-green-200">
            <div className="text-center">
              <p className="text-sm text-green-700 font-medium">Calculated BMI</p>
              <p className="text-2xl font-bold text-green-800">{calculateBMI()}</p>
              <p className="text-xs text-green-600">
                Height: {height}cm, Weight: {weight}kg
              </p>
            </div>
          </div>
        )}

        {/* Navigation Buttons */}
        <div className="flex justify-between pt-6">
          <button
            type="button"
            onClick={prevStep}
            disabled={currentStep === 1}
            className={`px-6 py-2 rounded-lg font-medium transition-colors duration-200 ${
              currentStep === 1
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'btn-secondary'
            }`}
          >
            Previous
          </button>

          {currentStep < totalSteps ? (
            <button
              type="button"
              onClick={nextStep}
              className="btn-primary"
            >
              Next
            </button>
          ) : (
            <button
              type="submit"
              className="btn-primary"
            >
              Get Health Assessment
            </button>
          )}
        </div>
      </form>

      {/* Help Text */}
      <div className="mt-8 p-4 bg-blue-50 rounded-lg">
        <div className="flex items-start space-x-3">
          <Shield className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-blue-800">
            <p className="font-medium mb-1">Privacy & Accuracy</p>
            <p>
              Your data is processed securely and anonymously. For best results, 
              provide accurate measurements from recent medical checkups. 
              This assessment is for educational purposes and should not replace 
              professional medical advice.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HealthForm;
