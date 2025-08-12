import React, { createContext, useContext, useReducer, useEffect } from "react";
import axios from "axios";

// Use Vite env at build time; fallback to local during development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const HealthContext = createContext();

const initialState = {
  patientData: null,
  healthAssessment: null,
  recommendations: null,
  loading: false,
  error: null,
  predictions: null,
  clusters: null,
  models: null
};

function healthReducer(state, action) {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload, error: null };
    case 'SET_ERROR':
      return { ...state, loading: false, error: action.payload };
    case 'SET_HEALTH_ASSESSMENT':
      return { ...state, loading: false, healthAssessment: action.payload, error: null };
    case 'SET_CLUSTERS':
      return { ...state, loading: false, clusters: action.payload, error: null };
    case 'SET_MODELS':
      return { ...state, models: action.payload };
    case 'CLEAR_DATA':
      return { ...initialState };
    default:
      return state;
  }
}

export const HealthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(healthReducer, initialState);

  // API functions
  const api = {
    predictHealthRisk: async (patientData) => {
      try {
        dispatch({ type: 'SET_LOADING', payload: true });
        const response = await axios.post(`${API_BASE_URL}/predict/health-risk`, {
          patient_data: patientData,
          model_type: 'random_forest'
        });
        
        dispatch({ type: 'SET_HEALTH_ASSESSMENT', payload: response.data });
        return response.data;
      } catch (error) {
        const errorMessage = error.response?.data?.detail || error.message;
        dispatch({ type: 'SET_ERROR', payload: errorMessage });
        throw error;
      }
    },

    clusterPatients: async (patientDataList) => {
      try {
        dispatch({ type: 'SET_LOADING', payload: true });
        const response = await axios.post(`${API_BASE_URL}/cluster/patients`, patientDataList);
        
        dispatch({ type: 'SET_CLUSTERS', payload: response.data });
        return response.data;
      } catch (error) {
        const errorMessage = error.response?.data?.detail || error.message;
        dispatch({ type: 'SET_ERROR', payload: errorMessage });
        throw error;
      }
    },

    getAvailableModels: async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/models/available`);
        dispatch({ type: 'SET_MODELS', payload: response.data });
        return response.data;
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    },

    getHealthParameters: async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/features/health-parameters`);
        return response.data;
      } catch (error) {
        console.error('Error fetching health parameters:', error);
      }
    },

    getRecommendationTemplates: async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/recommendations/templates`);
        return response.data;
      } catch (error) {
        console.error('Error fetching recommendation templates:', error);
      }
    }
  };

  // Load available models on component mount
  useEffect(() => {
    api.getAvailableModels();
  }, []);

  const value = {
    ...state,
    api,
    dispatch
  };

  return (
    <HealthContext.Provider value={value}>
      {children}
    </HealthContext.Provider>
  );
};

export const useHealth = () => {
  const context = useContext(HealthContext);
  if (!context) {
    throw new Error('useHealth must be used within a HealthProvider');
  }
  return context;
};
