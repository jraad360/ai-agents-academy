import React from 'react';
import { ArrowLeft } from 'lucide-react';
import { Module } from '../types';
import { CodeComparison } from './CodeComparison';

interface ModuleContentProps {
  module: Module;
  onBack: () => void;
}

export function ModuleContent({ module, onBack }: ModuleContentProps) {
  return (
    <div className="mt-4">
      <button
        onClick={onBack}
        className="flex items-center text-editor-accent hover:text-editor-accent/80 mb-6 font-mono"
      >
        <ArrowLeft size={20} className="mr-2" />
        cd ..
      </button>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        <h1 className="text-4xl font-mono font-bold text-editor-text dark:text-gray-100 mb-4">
          {module.title}
        </h1>
        
        <div className="flex items-center space-x-4 mb-8">
          <div className="tech-label">
            <span className="text-editor-text dark:text-gray-400">runtime:</span> {module.duration}
          </div>
          <div className="tech-label">
            <span className="text-editor-text dark:text-gray-400">level:</span> {module.level}
          </div>
        </div>

        {module.content?.sections.map((section, index) => (
          <div key={index} className="mt-12">
            <div className="terminal-card">
              <div className="terminal-header">
                <div className="terminal-dots">
                  <div className="terminal-dot terminal-dot-red"></div>
                  <div className="terminal-dot terminal-dot-yellow"></div>
                  <div className="terminal-dot terminal-dot-green"></div>
                </div>
                <h2 className="ml-4 text-2xl font-mono font-bold text-editor-text dark:text-gray-100">
                  {section.title}
                </h2>
              </div>
              <div className="p-6">
                <div className="text-editor-text dark:text-gray-300 whitespace-pre-wrap font-sans">
                  {section.content}
                </div>
                
                {section.codeExamples?.map((example, exampleIndex) => (
                  <CodeComparison key={exampleIndex} example={example} />
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}