// Test script to verify form data structure
// This simulates what the frontend sends to the backend

const testFormData = {
  age: 35,
  gender: "male",
  height: 175,
  weight: 70,
  blood_pressure: 120,
  glucose: 100,
  cholesterol: 180,
  creatinine: 1.0,
  smoking: "never_smoker",
  alcohol: "minimal",
  exercise_frequency: "weekly",
  sleep_hours: 7,
  stress_level: "low"
};

// Simulate the form processing
const processFormData = (data) => {
  // Calculate BMI
  const heightInMeters = data.height / 100;
  const bmi = (data.weight / (heightInMeters * heightInMeters)).toFixed(1);
  
  // Create the data structure the backend expects
  const processedData = {
    age: data.age || 30,
    gender: data.gender || 'male',
    blood_pressure: data.blood_pressure || 120,
    glucose: data.glucose || 100,
    bmi: parseFloat(bmi) || 25,
    cholesterol: data.cholesterol || 180,
    creatinine: data.creatinine || 1.0,
    smoking: data.smoking || 'never_smoker',
    alcohol: data.alcohol || 'minimal',
    sleep_hours: data.sleep_hours || 7,
    stress_level: data.stress_level || 'low',
    exercise_frequency: data.exercise_frequency || 'weekly'
  };
  
  return processedData;
};

// Test the processing
const result = processFormData(testFormData);
console.log("Original form data:", testFormData);
console.log("Processed data for backend:", result);
console.log("BMI calculated:", result.bmi);

// Verify all required fields are present
const requiredFields = [
  'age', 'gender', 'blood_pressure', 'glucose', 'bmi', 
  'cholesterol', 'creatinine', 'smoking', 'alcohol', 
  'sleep_hours', 'stress_level', 'exercise_frequency'
];

const missingFields = requiredFields.filter(field => !(field in result));
console.log("Missing fields:", missingFields.length === 0 ? "None" : missingFields);

if (missingFields.length === 0) {
  console.log("✅ All required fields are present!");
} else {
  console.log("❌ Missing required fields:", missingFields);
}
