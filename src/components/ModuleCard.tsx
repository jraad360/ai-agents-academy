import React from 'react';
import { Link } from 'react-router-dom';
import { Module } from '../types';
import { Clock, BookOpen } from 'lucide-react';

interface ModuleCardProps {
  module: Module;
  sectionId: string;
}

export function ModuleCard({ module, sectionId }: ModuleCardProps) {
  return (
    <Link 
      to={`/learn/${sectionId}/${module.id}`}
      className="block terminal-card group cursor-pointer transition-transform hover:-translate-y-1"
    >
      <div className="terminal-header">
        <div className="terminal-dots">
          <div className="terminal-dot terminal-dot-red"></div>
          <div className="terminal-dot terminal-dot-yellow"></div>
          <div className="terminal-dot terminal-dot-green"></div>
        </div>
      </div>
      
      <div className="p-6">
        <h3 className="text-xl font-mono font-semibold text-editor-text dark:text-gray-100">{module.title}</h3>
        <p className="mt-2 text-editor-muted dark:text-gray-400">{module.description}</p>
        
        <div className="mt-4 flex items-center space-x-4">
          <div className="tech-label">
            <Clock size={14} className="mr-1" />
            <span>{module.duration}</span>
          </div>
          <div className="tech-label">
            <BookOpen size={14} className="mr-1" />
            <span className="capitalize">{module.level}</span>
          </div>
        </div>
        
        <div className="mt-4 w-full px-4 py-2 bg-editor-accent bg-opacity-20 text-editor-accent rounded-md group-hover:bg-opacity-30 transition-all duration-200 text-center font-mono">
          ./start learn
        </div>
      </div>
    </Link>
  );
}