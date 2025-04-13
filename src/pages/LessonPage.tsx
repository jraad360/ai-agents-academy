import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ModuleContent } from '../components/ModuleContent';
import { sections } from '../data/sections';

export function LessonPage() {
  const { topicId, moduleId } = useParams();
  const navigate = useNavigate();

  const module = React.useMemo(() => {
    const section = sections.find(s => s.id === topicId);
    if (!section) return null;
    return section.modules.find(m => m.id === moduleId);
  }, [topicId, moduleId]);

  if (!module) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pt-24">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-editor-text dark:text-editor-text-dark">
            Lesson not found
          </h2>
          <button
            onClick={() => navigate(`/learn/${topicId}`)}
            className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-editor-accent hover:bg-editor-accent/90"
          >
            Return to Course
          </button>
        </div>
      </div>
    );
  }

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pt-24">
      <ModuleContent
        module={module}
        onBack={() => navigate(`/learn/${topicId}`)}
      />
    </main>
  );
}