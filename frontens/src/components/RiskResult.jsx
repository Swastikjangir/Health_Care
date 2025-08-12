import React from 'react';
import { 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  TrendingUp, 
  Heart, 
  Activity,
  Utensils,
  Calendar,
  Download,
  Share2
} from 'lucide-react';

const RiskResult = ({ assessment }) => {
  const { risk_level, risk_score, predictions, recommendations } = assessment;

  const getRiskColor = (level) => {
    switch (level.toLowerCase()) {
      case 'high risk':
        return 'danger';
      case 'medium risk':
        return 'warning';
      case 'low risk':
        return 'success';
      default:
        return 'info';
    }
  };

  const getRiskIcon = (level) => {
    switch (level.toLowerCase()) {
      case 'high risk':
        return AlertTriangle;
      case 'medium risk':
        return TrendingUp;
      case 'low risk':
        return CheckCircle;
      default:
        return Info;
    }
  };

  const RiskIcon = getRiskIcon(risk_level);
  const riskColor = getRiskColor(risk_level);

  const downloadReport = () => {
    const reportContent = `
HEALTH RISK ASSESSMENT REPORT
=============================

Overall Risk Level: ${risk_level}
Risk Score: ${risk_score}%

HEALTH ASSESSMENT:
${Object.entries(predictions.health_assessment || {}).map(([condition, assessment]) => 
  `• ${condition.replace('_', ' ').toUpperCase()}: ${assessment.risk.toUpperCase()} Risk`
).join('\n')}

RECOMMENDATIONS:
${Object.entries(recommendations || {}).map(([category, recs]) => 
  `${category.toUpperCase()}:\n${Array.isArray(recs) ? recs.map(rec => `  - ${rec}`).join('\n') : recs}`
).join('\n\n')}

NEXT STEPS:
${(predictions.next_steps || []).map(step => `• ${step}`).join('\n')}

Report generated on: ${new Date().toLocaleDateString()}
    `;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'health_assessment_report.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Overall Risk Summary */}
      <div className={`health-card ${riskColor}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className={`p-3 rounded-full bg-${riskColor === 'danger' ? 'red' : riskColor === 'warning' ? 'yellow' : 'green'}-100`}>
              <RiskIcon className={`h-8 w-8 text-${riskColor === 'danger' ? 'red' : riskColor === 'warning' ? 'yellow' : 'green'}-600`} />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-900">
                {risk_level}
              </h3>
              <p className="text-gray-600">
                Overall Health Risk Assessment
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-gray-900">
              {risk_score.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500">Risk Score</div>
          </div>
        </div>
      </div>

      {/* Risk Factors Breakdown */}
      <div className="card">
        <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center space-x-2">
          <Heart className="h-5 w-5 text-red-500" />
          <span>Risk Factors Analysis</span>
        </h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(predictions.health_assessment || {}).map(([condition, assessment]) => (
            <div key={condition} className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-gray-700">
                  {condition.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  assessment.risk === 'high' 
                    ? 'bg-red-100 text-red-800' 
                    : assessment.risk === 'medium'
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-green-100 text-green-800'
                }`}>
                  {assessment.risk.toUpperCase()}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                Risk Score: {assessment.score}/3
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommendations */}
      <div className="card">
        <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center space-x-2">
          <Activity className="h-5 w-5 text-green-500" />
          <span>Personalized Recommendations</span>
        </h3>
        
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Diet Recommendations */}
          {recommendations.diet && recommendations.diet.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800 flex items-center space-x-2">
                <Utensils className="h-4 w-4 text-orange-500" />
                <span>Diet & Nutrition</span>
              </h4>
              <ul className="space-y-2">
                {recommendations.diet.map((rec, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <span className="text-green-500 mt-1">•</span>
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Exercise Recommendations */}
          {recommendations.exercise && recommendations.exercise.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800 flex items-center space-x-2">
                <Activity className="h-4 w-4 text-blue-500" />
                <span>Exercise & Physical Activity</span>
              </h4>
              <ul className="space-y-2">
                {recommendations.exercise.map((rec, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <span className="text-blue-500 mt-1">•</span>
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Lifestyle Recommendations */}
          {recommendations.lifestyle && recommendations.lifestyle.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800 flex items-center space-x-2">
                <Heart className="h-4 w-4 text-purple-500" />
                <span>Lifestyle Changes</span>
              </h4>
              <ul className="space-y-2">
                {recommendations.lifestyle.map((rec, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <span className="text-purple-500 mt-1">•</span>
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Monitoring Recommendations */}
          {recommendations.monitoring && recommendations.monitoring.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800 flex items-center space-x-2">
                <Calendar className="h-4 w-4 text-indigo-500" />
                <span>Health Monitoring</span>
              </h4>
              <ul className="space-y-2">
                {recommendations.monitoring.map((rec, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <span className="text-indigo-500 mt-1">•</span>
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Next Steps */}
      {predictions.next_steps && predictions.next_steps.length > 0 && (
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-primary-500" />
            <span>Next Steps</span>
          </h3>
          <div className="space-y-3">
            {predictions.next_steps.map((step, index) => (
              <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                <span className="bg-primary-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-medium flex-shrink-0">
                  {index + 1}
                </span>
                <span className="text-gray-700">{step}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Follow-up Schedule */}
      {predictions.follow_up_schedule && (
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center space-x-2">
            <Calendar className="h-5 w-5 text-green-500" />
            <span>Follow-up Schedule</span>
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(predictions.follow_up_schedule).map(([period, timing]) => (
              <div key={period} className="text-center p-4 border rounded-lg">
                <div className="text-sm text-gray-500 mb-1">
                  {period.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </div>
                <div className="font-semibold text-gray-900">{timing}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <button
          onClick={downloadReport}
          className="btn-primary flex items-center justify-center space-x-2"
        >
          <Download className="h-4 w-4" />
          <span>Download Report</span>
        </button>
        <button className="btn-secondary flex items-center justify-center space-x-2">
          <Share2 className="h-4 w-4" />
          <span>Share Results</span>
        </button>
      </div>

      {/* Disclaimer */}
      <div className="text-center p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <div className="flex items-center justify-center space-x-2 mb-2">
          <Info className="h-5 w-5 text-yellow-600" />
          <span className="font-medium text-yellow-800">Important Disclaimer</span>
        </div>
        <p className="text-sm text-yellow-700">
          This health assessment is for educational and informational purposes only. 
          It is not a substitute for professional medical advice, diagnosis, or treatment. 
          Always consult with qualified healthcare professionals for medical concerns.
        </p>
      </div>
    </div>
  );
};

export default RiskResult;
