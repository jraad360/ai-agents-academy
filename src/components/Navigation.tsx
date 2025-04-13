import React from 'react';
import { Menu, Moon, Sun, Terminal } from 'lucide-react';
import { Theme } from '../types';
import { Link, useLocation } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

interface NavigationProps {
  theme: Theme;
}

export function Navigation({ theme }: NavigationProps) {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);
  const location = useLocation();

  const getPageTitle = () => {
    switch (location.pathname) {
      case '/':
        return 'AI Agent Academy - Learn AI Agent Development';
      case '/learn':
        return 'Learning Paths - AI Agent Academy';
      default:
        if (location.pathname.startsWith('/learn/')) {
          return 'Course Content - AI Agent Academy';
        }
        return 'AI Agent Academy';
    }
  };

  const getPageDescription = () => {
    switch (location.pathname) {
      case '/':
        return 'Master AI agent development with comprehensive tutorials covering multiple frameworks. Learn fundamentals and advanced patterns in TypeScript and Python.';
      case '/learn':
        return 'Explore our learning paths for AI agent development. Choose from LangChain, Google ADK, and more frameworks.';
      default:
        return 'Learn AI agent development with hands-on tutorials and practical examples.';
    }
  };

  return (
    <>
      <Helmet>
        <title>{getPageTitle()}</title>
        <meta name="description" content={getPageDescription()} />
      </Helmet>

      <nav aria-label="Main navigation" className="fixed top-0 w-full bg-editor-bg dark:bg-editor-bg-dark border-b border-editor-highlight dark:border-editor-highlight-dark z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center space-x-2">
              <Link to="/" className="flex items-center space-x-2" aria-label="Home">
                <Terminal className="text-editor-accent" size={24} />
                <span className="text-xl font-mono font-bold text-editor-text dark:text-gray-100">
                  AI Agent Academy
                </span>
              </Link>
            </div>

            <div className="hidden md:flex items-center">
              <nav className="tab-nav" aria-label="Secondary navigation">
                <Link 
                  to="/learn"
                  className={`tab-item ${location.pathname === '/learn' ? 'active' : ''}`}
                  aria-current={location.pathname === '/learn' ? 'page' : undefined}
                >
                  learn
                </Link>
              </nav>
              
              <button
                onClick={theme.toggleTheme}
                className="ml-4 p-2 rounded-lg bg-editor-highlight dark:bg-editor-highlight-dark text-editor-text dark:text-gray-100 hover:bg-opacity-80 transition-colors"
                aria-label={theme.isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {theme.isDark ? <Sun size={20} /> : <Moon size={20} />}
              </button>
            </div>

            <div className="md:hidden flex items-center">
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="p-2 text-editor-text dark:text-gray-100"
                aria-expanded={isMenuOpen}
                aria-label="Toggle menu"
              >
                <Menu size={24} />
              </button>
            </div>
          </div>
        </div>

        {isMenuOpen && (
          <div className="md:hidden bg-editor-bg dark:bg-editor-bg-dark border-t border-editor-highlight dark:border-editor-highlight-dark">
            <div className="px-2 pt-2 pb-3 space-y-1">
              <Link
                to="/learn"
                className="block w-full text-left px-3 py-2 text-editor-text dark:text-gray-300 hover:bg-editor-highlight dark:hover:bg-editor-highlight-dark rounded-md font-mono"
                onClick={() => setIsMenuOpen(false)}
              >
                ./learn
              </Link>
            </div>
          </div>
        )}
      </nav>
    </>
  );
}