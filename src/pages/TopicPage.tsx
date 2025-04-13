import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ModuleCard } from '../components/ModuleCard';
import { ArrowLeft } from 'lucide-react';
import { sections } from '../data/sections';

const setupGuides = {
  langchain: {
    prerequisites: [
      '• Python 3.9+ installed on your system',
      '• Node.js 18+ for TypeScript development',
      '• A code editor (VS Code recommended)',
      '• Basic knowledge of async/await and type systems'
    ],
    installation: {
      python: 'pip install langchain langchain-core langgraph langchain-openai langchain-community',
      typescript: 'npm install @langchain/core @langchain/openai @langchain/community langgraph-ts zod'
    },
    resources: [
      {
        title: 'LangChain Documentation',
        url: 'https://python.langchain.com/docs/get_started/introduction'
      },
      {
        title: 'LCEL Documentation',
        url: 'https://python.langchain.com/docs/expression_language/interface'
      },
      {
        title: 'LangGraph Documentation',
        url: 'https://python.langchain.com/docs/langgraph'
      }
    ]
  },
  'google-adk': {
    prerequisites: [
      '• Python 3.9+ installed on your system',
      '• Google Cloud account with billing enabled',
      '• Google Cloud CLI installed',
      '• Basic knowledge of Python and async programming'
    ],
    installation: {
      python: 'pip install google-adk google-cloud-aiplatform',
      typescript: 'npm install @google-cloud/adk @google-cloud/aiplatform'
    },
    resources: [
      {
        title: 'Google ADK Documentation',
        url: 'https://developers.generativeai.google/guide/adk_overview'
      },
      {
        title: 'Google Cloud Console',
        url: 'https://console.cloud.google.com'
      }
    ]
  }
};

export function TopicPage() {
  const { topicId } = useParams();
  const navigate = useNavigate();
  
  const section = React.useMemo(() => {
    return sections.find(s => s.id === topicId);
  }, [topicId]);

  const setup = topicId ? setupGuides[topicId as keyof typeof setupGuides] : null;

  if (!section || !setup) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pt-24">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-editor-text dark:text-editor-text-dark">
            Topic not found
          </h2>
          <button
            onClick={() => navigate('/learn')}
            className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-editor-accent hover:bg-editor-accent/90"
          >
            Return to Learn
          </button>
        </div>
      </div>
    );
  }

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pt-24">
      <button
        onClick={() => navigate('/learn')}
        className="flex items-center text-editor-accent hover:text-editor-accent/80 mb-6 font-mono"
      >
        <ArrowLeft size={20} className="mr-2" />
        cd ..
      </button>

      <div className="terminal-card p-6 mb-8">
        <h1 className="text-4xl font-bold text-editor-text dark:text-white">
          {section.title}
        </h1>
        <p className="mt-2 text-xl text-editor-muted dark:text-gray-400">
          {section.description}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {section.modules.map((module) => (
          <ModuleCard
            key={module.id}
            module={module}
            sectionId={section.id}
          />
        ))}
      </div>

      <section id="setup" className="mt-16">
        <div className="terminal-card p-6">
          <h2 className="text-3xl font-bold text-editor-text dark:text-white">Setup Guide</h2>
          <p className="mt-2 text-editor-muted dark:text-gray-400">
            Get your development environment ready for {section.title}
          </p>
          
          <div className="mt-8">
            <h3 className="text-xl font-semibold text-editor-text dark:text-white mb-4">Prerequisites</h3>
            <ul className="space-y-3 text-editor-muted dark:text-gray-400">
              {setup.prerequisites.map((prereq, index) => (
                <li key={index}>{prereq}</li>
              ))}
            </ul>
            
            <h3 className="text-xl font-semibold text-editor-text dark:text-white mt-8 mb-4">Installation</h3>
            <div className="bg-editor-bg dark:bg-editor-bg-dark rounded p-4">
              <p className="text-sm text-editor-muted dark:text-gray-400 mb-2">Python Setup:</p>
              <pre className="bg-editor-highlight dark:bg-editor-highlight-dark p-3 rounded">
                <code>{setup.installation.python}</code>
              </pre>
              
              <p className="text-sm text-editor-muted dark:text-gray-400 mt-4 mb-2">TypeScript Setup:</p>
              <pre className="bg-editor-highlight dark:bg-editor-highlight-dark p-3 rounded">
                <code>{setup.installation.typescript}</code>
              </pre>
            </div>

            <div className="mt-8">
              <h3 className="text-xl font-semibold text-editor-text dark:text-white mb-4">Additional Resources</h3>
              <ul className="space-y-2">
                {setup.resources.map((resource, index) => (
                  <li key={index}>
                    <a 
                      href={resource.url}
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-editor-accent hover:text-editor-accent/80"
                    >
                      • {resource.title}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}