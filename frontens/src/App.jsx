import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Prediction from './pages/Prediction';
import About from './pages/About';
import { HealthProvider } from './context/HealthContext';

function App() {
  return (
    <Router>
      <HealthProvider>
        <div className="min-h-screen bg-gray-50">
          <Navbar />
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/prediction" element={<Prediction />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </main>
        </div>
      </HealthProvider>
    </Router>
  );
}

export default App;
