import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, Code2, BookOpen } from 'lucide-react';
import { sections } from '../data/sections';

export function Learn() {
  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pt-24">
      <h1 className="text-4xl font-bold text-editor-text dark:text-white mb-6">Learning Paths</h1>
      <p className="text-xl text-editor-muted dark:text-gray-400 mb-12">
        Choose a learning path to begin your journey into AI agent development
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {sections.map((section) => (
          <Link
            key={section.id}
            to={`/learn/${section.id}`}
            className="terminal-card p-6 hover:scale-[1.02] transition-transform duration-200"
          >
            <div className="flex items-start space-x-4">
              <div className="bg-editor-accent bg-opacity-10 p-3 rounded-lg">
                <Brain className="h-8 w-8 text-editor-accent" />
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-editor-text dark:text-white mb-2">
                  {section.title}
                </h2>
                <p className="text-editor-muted dark:text-gray-400 mb-4">
                  {section.description}
                </p>
                <div className="flex items-center space-x-4">
                  <div className="tech-label">
                    <BookOpen size={14} className="mr-1" />
                    <span>{section.modules.length} Modules</span>
                  </div>
                  <div className="tech-label">
                    <Code2 size={14} className="mr-1" />
                    <span>TypeScript & Python</span>
                  </div>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </main>
  );
}