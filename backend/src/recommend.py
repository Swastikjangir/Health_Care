import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json

class HealthRecommendationEngine:
    def __init__(self):
        self.recommendation_rules = self._initialize_recommendation_rules()
        self.diet_recommendations = self._initialize_diet_recommendations()
        self.exercise_recommendations = self._initialize_exercise_recommendations()
        self.lifestyle_recommendations = self._initialize_lifestyle_recommendations()
        
    def _initialize_recommendation_rules(self) -> Dict:
        """Initialize recommendation rules based on health parameters"""
        return {
            'diabetes': {
                'glucose': {
                    'high': {'threshold': 140, 'risk': 'high', 'priority': 1},
                    'medium': {'threshold': 100, 'risk': 'medium', 'priority': 2},
                    'normal': {'threshold': 70, 'risk': 'low', 'priority': 3}
                },
                'bmi': {
                    'high': {'threshold': 30, 'risk': 'high', 'priority': 1},
                    'medium': {'threshold': 25, 'risk': 'medium', 'priority': 2},
                    'normal': {'threshold': 18.5, 'risk': 'low', 'priority': 3}
                }
            },
            'heart_disease': {
                'blood_pressure': {
                    'high': {'threshold': 140, 'risk': 'high', 'priority': 1},
                    'medium': {'threshold': 120, 'risk': 'medium', 'priority': 2},
                    'normal': {'threshold': 90, 'risk': 'low', 'priority': 3}
                },
                'cholesterol': {
                    'high': {'threshold': 240, 'risk': 'high', 'priority': 1},
                    'medium': {'threshold': 200, 'risk': 'medium', 'priority': 2},
                    'normal': {'threshold': 150, 'risk': 'low', 'priority': 3}
                }
            },
            'kidney_disease': {
                'creatinine': {
                    'high': {'threshold': 1.5, 'risk': 'high', 'priority': 1},
                    'medium': {'threshold': 1.2, 'risk': 'medium', 'priority': 2},
                    'normal': {'threshold': 0.8, 'risk': 'low', 'priority': 3}
                }
            }
        }
    
    def _initialize_diet_recommendations(self) -> Dict:
        """Initialize diet recommendations"""
        return {
            'diabetes': {
                'high_risk': [
                    "Limit carbohydrate intake to 45-60g per meal",
                    "Choose low glycemic index foods",
                    "Increase fiber intake (25-30g daily)",
                    "Avoid sugary beverages and processed foods",
                    "Eat regular meals at consistent times"
                ],
                'medium_risk': [
                    "Moderate carbohydrate intake",
                    "Include more whole grains and vegetables",
                    "Limit added sugars",
                    "Monitor portion sizes"
                ],
                'low_risk': [
                    "Maintain balanced diet",
                    "Include variety of fruits and vegetables",
                    "Choose whole grains over refined grains"
                ]
            },
            'heart_disease': {
                'high_risk': [
                    "Reduce sodium intake to <1500mg daily",
                    "Limit saturated fats to <7% of calories",
                    "Increase omega-3 fatty acids",
                    "Eat more fruits, vegetables, and whole grains",
                    "Avoid trans fats completely"
                ],
                'medium_risk': [
                    "Moderate sodium intake",
                    "Choose lean proteins",
                    "Include heart-healthy fats",
                    "Limit processed foods"
                ],
                'low_risk': [
                    "Maintain heart-healthy diet",
                    "Regular consumption of fish and nuts",
                    "Balanced macronutrient intake"
                ]
            },
            'general_health': {
                'high_risk': [
                    "Consult registered dietitian",
                    "Strict dietary modifications",
                    "Regular monitoring of food intake",
                    "Consider medical nutrition therapy"
                ],
                'medium_risk': [
                    "Gradual dietary changes",
                    "Regular meal planning",
                    "Nutrition education"
                ],
                'low_risk': [
                    "Maintain current healthy eating habits",
                    "Regular health check-ups",
                    "Preventive nutrition"
                ]
            },
            'obesity': {
                'high_risk': [
                    "Calorie-controlled diet plan",
                    "High protein, moderate carb intake",
                    "Regular meal timing",
                    "Avoid late-night eating",
                    "Consult nutritionist for personalized plan"
                ],
                'medium_risk': [
                    "Moderate calorie reduction",
                    "Increase protein and fiber",
                    "Portion control strategies",
                    "Regular meal planning"
                ],
                'low_risk': [
                    "Maintain healthy weight",
                    "Balanced nutrition",
                    "Regular physical activity"
                ]
            },
            'age_risk': {
                'high_risk': [
                    "Age-appropriate nutrition",
                    "Increased protein intake",
                    "Vitamin D and calcium rich foods",
                    "Regular health monitoring",
                    "Consult geriatric nutritionist"
                ],
                'medium_risk': [
                    "Balanced aging nutrition",
                    "Regular health check-ups",
                    "Preventive nutrition focus"
                ],
                'low_risk': [
                    "Maintain healthy aging habits",
                    "Regular exercise and nutrition",
                    "Preventive care"
                ]
            },
            'sleep': {
                'poor': [
                    "Avoid caffeine after 2 PM",
                    "Limit heavy meals before bedtime",
                    "Include tryptophan-rich foods",
                    "Stay hydrated but avoid excess fluids",
                    "Consider herbal teas for relaxation"
                ],
                'adequate': [
                    "Maintain regular meal timing",
                    "Balanced nutrition throughout day",
                    "Include sleep-supporting nutrients"
                ]
            },
            'stress': {
                'high': [
                    "Include stress-reducing foods",
                    "Omega-3 rich foods",
                    "Complex carbohydrates for mood",
                    "Limit caffeine and sugar",
                    "Stay well-hydrated"
                ],
                'moderate': [
                    "Balanced stress-management nutrition",
                    "Regular meal timing",
                    "Include mood-supporting foods"
                ],
                'low': [
                    "Maintain healthy eating habits",
                    "Regular nutrition routine",
                    "Preventive nutrition focus"
                ]
            },
            'sleep': {
                'poor': [
                    "Avoid caffeine after 2 PM",
                    "Limit heavy meals before bedtime",
                    "Include tryptophan-rich foods",
                    "Stay hydrated but avoid excess fluids",
                    "Consider herbal teas for relaxation"
                ],
                'adequate': [
                    "Maintain regular meal timing",
                    "Balanced nutrition throughout day",
                    "Include sleep-supporting nutrients"
                ]
            }
        }
    
    def _initialize_exercise_recommendations(self) -> Dict:
        """Initialize exercise recommendations"""
        return {
            'diabetes': {
                'high_risk': [
                    "150 minutes moderate aerobic exercise weekly",
                    "Strength training 2-3 times per week",
                    "Monitor blood glucose before/during/after exercise",
                    "Start with low-impact activities",
                    "Exercise under medical supervision initially"
                ],
                'medium_risk': [
                    "120 minutes moderate exercise weekly",
                    "Include both cardio and strength training",
                    "Monitor blood glucose response",
                    "Gradual intensity progression"
                ],
                'low_risk': [
                    "150 minutes moderate exercise weekly",
                    "Mix of aerobic and strength training",
                    "Regular physical activity maintenance"
                ]
            },
            'heart_disease': {
                'high_risk': [
                    "Cardiac rehabilitation program",
                    "Low-intensity walking (10-15 minutes)",
                    "Gradual progression under supervision",
                    "Avoid high-intensity activities",
                    "Regular medical clearance"
                ],
                'medium_risk': [
                    "Moderate aerobic exercise",
                    "Regular walking program",
                    "Include flexibility exercises",
                    "Monitor heart rate and symptoms"
                ],
                'low_risk': [
                    "Regular aerobic exercise",
                    "Strength training 2-3 times weekly",
                    "Flexibility and balance exercises"
                ]
            },
            'general_fitness': {
                'high_risk': [
                    "Start with walking 10-15 minutes daily",
                    "Gradual progression under guidance",
                    "Focus on low-impact activities",
                    "Regular medical monitoring"
                ],
                'medium_risk': [
                    "Moderate exercise program",
                    "Mix of cardio and strength",
                    "Regular activity schedule"
                ],
                'low_risk': [
                    "Maintain current fitness routine",
                    "Variety in exercise types",
                    "Regular physical activity"
                ]
            },
            'obesity': {
                'high_risk': [
                    "Low-impact walking (20-30 minutes daily)",
                    "Swimming or water aerobics",
                    "Gradual progression under supervision",
                    "Focus on consistency over intensity",
                    "Regular medical monitoring"
                ],
                'medium_risk': [
                    "Moderate walking program",
                    "Low-impact cardio exercises",
                    "Basic strength training",
                    "Regular activity schedule"
                ],
                'low_risk': [
                    "Regular cardio and strength training",
                    "Variety in exercise types",
                    "Maintain healthy weight through activity"
                ]
            },
            'age_risk': {
                'high_risk': [
                    "Gentle walking program",
                    "Chair exercises and stretching",
                    "Balance and flexibility training",
                    "Supervised exercise program",
                    "Regular medical clearance"
                ],
                'medium_risk': [
                    "Moderate walking and stretching",
                    "Light strength training",
                    "Balance exercises",
                    "Regular activity maintenance"
                ],
                'low_risk': [
                    "Regular exercise routine",
                    "Strength and flexibility training",
                    "Active lifestyle maintenance"
                ]
            },
            'sleep': {
                'poor': [
                    "Gentle evening exercises",
                    "Yoga and stretching",
                    "Avoid intense exercise before bed",
                    "Focus on relaxation techniques"
                ],
                'adequate': [
                    "Maintain regular exercise routine",
                    "Include relaxation exercises",
                    "Consistent activity schedule"
                ]
            },
            'stress': {
                'high': [
                    "Gentle walking and stretching",
                    "Yoga and meditation exercises",
                    "Low-impact activities",
                    "Focus on stress relief through movement"
                ],
                'moderate': [
                    "Regular moderate exercise",
                    "Include stress-relief activities",
                    "Balanced workout routine"
                ],
                'low': [
                    "Maintain current exercise routine",
                    "Include variety in activities",
                    "Regular physical activity"
                ]
            }
        }
    
    def _initialize_lifestyle_recommendations(self) -> Dict:
        """Initialize lifestyle recommendations"""
        return {
            'smoking': {
                'current_smoker': [
                    "Immediate smoking cessation program",
                    "Nicotine replacement therapy if needed",
                    "Behavioral counseling",
                    "Support group participation",
                    "Regular follow-up with healthcare provider"
                ],
                'former_smoker': [
                    "Maintain smoke-free status",
                    "Avoid secondhand smoke exposure",
                    "Regular health monitoring",
                    "Lung health maintenance"
                ],
                'never_smoker': [
                    "Continue avoiding tobacco products",
                    "Support others in quitting",
                    "Regular health check-ups"
                ]
            },
            'alcohol': {
                'excessive': [
                    "Reduce alcohol consumption",
                    "Seek professional help if needed",
                    "Set specific reduction goals",
                    "Monitor consumption patterns"
                ],
                'moderate': [
                    "Maintain moderate consumption",
                    "Alcohol-free days",
                    "Monitor health effects"
                ],
                'minimal': [
                    "Continue current pattern",
                    "Regular health monitoring"
                ]
            },
            'sleep': {
                'poor': [
                    "Establish regular sleep schedule",
                    "Create sleep-conducive environment",
                    "Limit screen time before bed",
                    "Consider sleep hygiene consultation"
                ],
                'adequate': [
                    "Maintain current sleep habits",
                    "Regular sleep schedule",
                    "Quality sleep monitoring"
                ]
            },
            'stress': {
                'high_risk': [
                    "Stress management techniques",
                    "Regular relaxation exercises",
                    "Consider counseling or therapy",
                    "Work-life balance improvement"
                ],
                'medium_risk': [
                    "Regular stress relief activities",
                    "Mindfulness practices",
                    "Time management skills"
                ],
                'low_risk': [
                    "Maintain current stress management",
                    "Regular wellness activities"
                ]
            }
        }
    
    def assess_health_parameters(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess health parameters and assign risk levels"""
        assessment = {}
        
        # Diabetes risk assessment
        if 'glucose' in patient_data:
            glucose = patient_data['glucose']
            if glucose >= 140:
                assessment['diabetes'] = {'risk': 'high_risk', 'score': 3}
            elif glucose >= 100:
                assessment['diabetes'] = {'risk': 'medium_risk', 'score': 2}
            else:
                assessment['diabetes'] = {'risk': 'low_risk', 'score': 1}
        
        # Heart disease risk assessment
        if 'blood_pressure' in patient_data:
            bp = patient_data['blood_pressure']
            if bp >= 140:
                assessment['heart_disease'] = {'risk': 'high_risk', 'score': 3}
            elif bp >= 120:
                assessment['heart_disease'] = {'risk': 'medium_risk', 'score': 2}
            else:
                assessment['heart_disease'] = {'risk': 'low_risk', 'score': 1}
        
        # BMI assessment
        if 'bmi' in patient_data:
            bmi = patient_data['bmi']
            if bmi >= 30:
                assessment['obesity'] = {'risk': 'high_risk', 'score': 3}
            elif bmi >= 25:
                assessment['obesity'] = {'risk': 'medium_risk', 'score': 2}
            else:
                assessment['obesity'] = {'risk': 'low_risk', 'score': 1}
        
        # Age-based risk
        if 'age' in patient_data:
            age = patient_data['age']
            if age >= 65:
                assessment['age_risk'] = {'risk': 'high_risk', 'score': 3}
            elif age >= 45:
                assessment['age_risk'] = {'risk': 'medium_risk', 'score': 2}
            else:
                assessment['age_risk'] = {'risk': 'low_risk', 'score': 1}
        
        # Sleep assessment
        if 'sleep_hours' in patient_data:
            sleep_hours = patient_data['sleep_hours']
            if sleep_hours < 6:
                assessment['sleep'] = {'risk': 'poor', 'score': 3}
            elif sleep_hours < 7:
                assessment['sleep'] = {'risk': 'poor', 'score': 2}
            else:
                assessment['sleep'] = {'risk': 'adequate', 'score': 1}
        
        # Stress assessment
        if 'stress_level' in patient_data:
            stress_level = patient_data['stress_level']
            if stress_level in ['high', 'High', 'HIGH']:
                assessment['stress'] = {'risk': 'high_risk', 'score': 3}
            elif stress_level in ['medium', 'Medium', 'MEDIUM']:
                assessment['stress'] = {'risk': 'medium_risk', 'score': 2}
            else:
                assessment['stress'] = {'risk': 'low_risk', 'score': 1}
        
        return assessment
    
    def generate_recommendations(self, patient_data: Dict[str, Any], 
                               health_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive health recommendations"""
        recommendations = {
            'diet': [],
            'exercise': [],
            'lifestyle': [],
            'monitoring': [],
            'priority_actions': []
        }
        
        # Generate diet recommendations
        for condition, assessment in health_assessment.items():
            if condition in self.diet_recommendations:
                risk_level = assessment['risk']
                if risk_level in self.diet_recommendations[condition]:
                    recommendations['diet'].extend(
                        self.diet_recommendations[condition][risk_level]
                    )
        
        # Generate exercise recommendations
        for condition, assessment in health_assessment.items():
            if condition in self.exercise_recommendations:
                risk_level = assessment['risk']
                if risk_level in self.exercise_recommendations[condition]:
                    recommendations['exercise'].extend(
                        self.exercise_recommendations[condition][risk_level]
                    )
        
        # Generate lifestyle recommendations
        if 'smoking' in patient_data:
            smoking_status = patient_data['smoking']
            if smoking_status in self.lifestyle_recommendations['smoking']:
                recommendations['lifestyle'].extend(
                    self.lifestyle_recommendations['smoking'][smoking_status]
                )
        
        if 'alcohol' in patient_data:
            alcohol_level = patient_data['alcohol']
            if alcohol_level in self.lifestyle_recommendations['alcohol']:
                recommendations['lifestyle'].extend(
                    self.lifestyle_recommendations['alcohol'][alcohol_level]
                )
        
        # Add sleep recommendations
        if 'sleep_hours' in patient_data:
            sleep_hours = patient_data['sleep_hours']
            if sleep_hours < 7:
                recommendations['lifestyle'].extend(self.lifestyle_recommendations['sleep']['poor'])
            else:
                recommendations['lifestyle'].extend(self.lifestyle_recommendations['sleep']['adequate'])
        
        # Add stress recommendations
        if 'stress_level' in patient_data:
            stress_level = patient_data['stress_level']
            if stress_level in ['high', 'High', 'HIGH']:
                recommendations['lifestyle'].extend(self.lifestyle_recommendations['stress']['high_risk'])
            elif stress_level in ['medium', 'Medium', 'MEDIUM']:
                recommendations['lifestyle'].extend(self.lifestyle_recommendations['stress']['medium_risk'])
            else:
                recommendations['lifestyle'].extend(self.lifestyle_recommendations['stress']['low_risk'])
        
        # Add default recommendations if none were generated
        if not recommendations['diet']:
            recommendations['diet'] = [
                "Maintain a balanced diet with fruits and vegetables",
                "Stay hydrated throughout the day",
                "Limit processed foods and added sugars"
            ]
        
        if not recommendations['exercise']:
            recommendations['exercise'] = [
                "Aim for 150 minutes of moderate exercise weekly",
                "Include both cardio and strength training",
                "Stay active throughout the day"
            ]
        
        if not recommendations['lifestyle']:
            recommendations['lifestyle'] = [
                "Maintain regular sleep schedule",
                "Practice stress management techniques",
                "Regular health check-ups"
            ]
        
        # Generate monitoring recommendations
        recommendations['monitoring'] = self._generate_monitoring_plan(health_assessment)
        
        # Generate priority actions
        recommendations['priority_actions'] = self._generate_priority_actions(
            health_assessment, patient_data
        )
        
        return recommendations
    
    def _generate_monitoring_plan(self, health_assessment: Dict[str, Any]) -> List[str]:
        """Generate monitoring plan based on risk levels"""
        monitoring_plan = []
        
        # Determine overall risk level
        total_score = sum(assessment['score'] for assessment in health_assessment.values())
        
        if total_score >= 8:
            monitoring_plan.extend([
                "Weekly health monitoring",
                "Monthly doctor visits",
                "Daily symptom tracking",
                "Regular lab tests as prescribed"
            ])
        elif total_score >= 5:
            monitoring_plan.extend([
                "Bi-weekly health monitoring",
                "Quarterly doctor visits",
                "Weekly symptom tracking",
                "Annual comprehensive health check"
            ])
        else:
            monitoring_plan.extend([
                "Monthly health monitoring",
                "Annual doctor visits",
                "Regular preventive screenings"
            ])
        
        return monitoring_plan
    
    def _generate_priority_actions(self, health_assessment: Dict[str, Any], 
                                 patient_data: Dict[str, Any]) -> List[str]:
        """Generate priority actions based on highest risk factors"""
        priority_actions = []
        
        # Sort conditions by risk score
        sorted_conditions = sorted(
            health_assessment.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        # Generate actions for top 3 highest risk conditions
        for i, (condition, assessment) in enumerate(sorted_conditions[:3]):
            if condition == 'diabetes' and assessment['risk'] == 'high_risk':
                priority_actions.append("Immediate blood glucose monitoring and consultation")
            elif condition == 'heart_disease' and assessment['risk'] == 'high_risk':
                priority_actions.append("Cardiac evaluation and blood pressure management")
            elif condition == 'obesity' and assessment['risk'] == 'high_risk':
                priority_actions.append("Weight management program and nutrition consultation")
        
        # Add general priority actions
        if any(assessment['risk'] == 'high_risk' for assessment in health_assessment.values()):
            priority_actions.append("Schedule immediate medical consultation")
        
        return priority_actions
    
    def create_personalized_plan(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete personalized health plan"""
        # Assess health parameters
        health_assessment = self.assess_health_parameters(patient_data)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(patient_data, health_assessment)
        
        # Calculate overall risk score
        total_risk_score = sum(assessment['score'] for assessment in health_assessment.values())
        max_possible_score = len(health_assessment) * 3
        
        overall_risk_percentage = (total_risk_score / max_possible_score) * 100
        
        # Determine overall risk level
        if overall_risk_percentage >= 70:
            overall_risk_level = "High Risk"
        elif overall_risk_percentage >= 40:
            overall_risk_level = "Medium Risk"
        else:
            overall_risk_level = "Low Risk"
        
        # Create personalized plan
        personalized_plan = {
            'patient_summary': {
                'overall_risk_level': overall_risk_level,
                'overall_risk_score': overall_risk_percentage,
                'health_assessment': health_assessment
            },
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(overall_risk_level),
            'follow_up_schedule': self._generate_follow_up_schedule(overall_risk_level)
        }
        
        return personalized_plan
    
    def _generate_next_steps(self, risk_level: str) -> List[str]:
        """Generate next steps based on risk level"""
        if risk_level == "High Risk":
            return [
                "Immediate medical consultation within 48 hours",
                "Emergency contact information review",
                "Medication review if applicable",
                "Lifestyle modification plan development"
            ]
        elif risk_level == "Medium Risk":
            return [
                "Schedule medical consultation within 1 week",
                "Begin implementing recommended changes",
                "Set up monitoring schedule",
                "Review and adjust plan in 2 weeks"
            ]
        else:
            return [
                "Schedule routine health check-up",
                "Continue preventive measures",
                "Monitor for any changes",
                "Annual comprehensive review"
            ]
    
    def _generate_follow_up_schedule(self, risk_level: str) -> Dict[str, str]:
        """Generate follow-up schedule based on risk level"""
        if risk_level == "High Risk":
            return {
                "immediate": "Within 48 hours",
                "short_term": "1 week",
                "medium_term": "1 month",
                "long_term": "3 months"
            }
        elif risk_level == "Medium Risk":
            return {
                "immediate": "Within 1 week",
                "short_term": "2 weeks",
                "medium_term": "1 month",
                "long_term": "3 months"
            }
        else:
            return {
                "immediate": "Within 1 month",
                "short_term": "3 months",
                "medium_term": "6 months",
                "long_term": "1 year"
            }
    
    def export_recommendations(self, personalized_plan: Dict[str, Any], 
                             format_type: str = 'json') -> str:
        """Export recommendations in specified format"""
        if format_type == 'json':
            return json.dumps(personalized_plan, indent=2)
        elif format_type == 'text':
            return self._format_as_text(personalized_plan)
        else:
            raise ValueError("Unsupported format type. Use 'json' or 'text'")
    
    def _format_as_text(self, personalized_plan: Dict[str, Any]) -> str:
        """Format recommendations as readable text"""
        text_output = "SMART HEALTH RECOMMENDATIONS\n"
        text_output += "=" * 50 + "\n\n"
        
        # Patient Summary
        summary = personalized_plan['patient_summary']
        text_output += f"OVERALL RISK LEVEL: {summary['overall_risk_level']}\n"
        text_output += f"RISK SCORE: {summary['overall_risk_score']:.1f}%\n\n"
        
        # Health Assessment
        text_output += "HEALTH ASSESSMENT:\n"
        for condition, assessment in summary['health_assessment'].items():
            text_output += f"- {condition.replace('_', ' ').title()}: {assessment['risk'].upper()} Risk\n"
        text_output += "\n"
        
        # Recommendations
        recommendations = personalized_plan['recommendations']
        text_output += "DIET RECOMMENDATIONS:\n"
        for rec in recommendations['diet']:
            text_output += f"• {rec}\n"
        text_output += "\n"
        
        text_output += "EXERCISE RECOMMENDATIONS:\n"
        for rec in recommendations['exercise']:
            text_output += f"• {rec}\n"
        text_output += "\n"
        
        text_output += "LIFESTYLE RECOMMENDATIONS:\n"
        for rec in recommendations['lifestyle']:
            text_output += f"• {rec}\n"
        text_output += "\n"
        
        # Next Steps
        text_output += "PRIORITY ACTIONS:\n"
        for action in recommendations['priority_actions']:
            text_output += f"• {action}\n"
        text_output += "\n"
        
        # Follow-up Schedule
        follow_up = personalized_plan['follow_up_schedule']
        text_output += "FOLLOW-UP SCHEDULE:\n"
        for period, timing in follow_up.items():
            text_output += f"• {period.replace('_', ' ').title()}: {timing}\n"
        
        return text_output
