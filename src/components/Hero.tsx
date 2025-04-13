import React from 'react';
import { Brain, Code2, BookOpen, ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';

export function Hero() {
  return (
    <div className="relative bg-editor-bg dark:bg-editor-bg-dark pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 text-center">
        <h1 className="text-4xl tracking-tight font-mono font-bold text-editor-text dark:text-gray-100 sm:text-5xl md:text-6xl">
          <span className="block">Master AI Agent Development</span>
          <span className="block text-editor-accent">with Modern Frameworks</span>
        </h1>
        <p className="mt-3 max-w-md mx-auto text-base text-editor-muted dark:text-gray-400 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
          A comprehensive guide for experienced software engineers diving into AI agent development.
          Learn multiple frameworks and approaches to building intelligent agents.
        </p>
        
        <div className="mt-10 flex justify-center">
          <Link 
            to="/learn" 
            className="inline-flex items-center px-6 py-3 bg-editor-accent bg-opacity-20 text-editor-accent rounded-md hover:bg-opacity-30 transition-all duration-200 font-mono"
          >
            ./start learn
            <ChevronRight className="ml-2" size={20} />
          </Link>
        </div>

        <div className="mt-20 grid grid-cols-1 gap-8 sm:grid-cols-3">
          <div className="terminal-card p-6">
            <Brain className="h-12 w-12 text-editor-accent mx-auto" />
            <h3 className="mt-4 text-xl font-mono font-semibold text-editor-text dark:text-gray-100">Multiple Frameworks</h3>
            <p className="mt-2 text-editor-muted dark:text-gray-400">Learn different approaches to building AI agents</p>
          </div>
          <div className="terminal-card p-6">
            <Code2 className="h-12 w-12 text-editor-accent mx-auto" />
            <h3 className="mt-4 text-xl font-mono font-semibold text-editor-text dark:text-gray-100">Practical Examples</h3>
            <p className="mt-2 text-editor-muted dark:text-gray-400">Ready-to-run code examples in TypeScript and Python</p>
          </div>
          <div className="terminal-card p-6">
            <BookOpen className="h-12 w-12 text-editor-accent mx-auto" />
            <h3 className="mt-4 text-xl font-mono font-semibold text-editor-text dark:text-gray-100">Best Practices</h3>
            <p className="mt-2 text-editor-muted dark:text-gray-400">Learn production-ready patterns and architectures</p>
          </div>
        </div>
      </div>
    </div>
  );
}