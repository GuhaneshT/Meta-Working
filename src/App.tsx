import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout';
import UploadPage from './pages/UploadPage';
import PreprocessPage from './pages/PreprocessPage';
import VisualizePage from './pages/VisualizePage';

function App() {
  return (
    <Router>
      <Toaster position="top-right" />
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/upload" replace />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="preprocess" element={<PreprocessPage />} />
          <Route path="visualize" element={<VisualizePage />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;