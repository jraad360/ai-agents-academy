import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';
import { Navigation } from './components/Navigation';
import { Home } from './pages/Home';
import { Learn } from './pages/Learn';
import { TopicPage } from './pages/TopicPage';
import { LessonPage } from './pages/LessonPage';

function App() {
  const [isDark, setIsDark] = React.useState(false);

  React.useEffect(() => {
    // Check system preference on mount
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDark(prefersDark);

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => setIsDark(e.matches);
    mediaQuery.addEventListener('change', handleChange);

    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  React.useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  const theme = {
    isDark,
    toggleTheme: () => setIsDark(!isDark)
  };

  return (
    <HelmetProvider>
      <Router>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
          <Navigation theme={theme} />
          
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/learn" element={<Learn />} />
            <Route path="/learn/:topicId" element={<TopicPage />} />
            <Route path="/learn/:topicId/:moduleId" element={<LessonPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Router>
    </HelmetProvider>
  );
}

export default App;