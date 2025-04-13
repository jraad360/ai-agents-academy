import React from 'react';
import { getHighlighter, Highlighter } from 'shiki';
import { CodeExample } from '../types';

interface CodeComparisonProps {
  example: CodeExample;
}

export function CodeComparison({ example }: CodeComparisonProps) {
  const [activeTab, setActiveTab] = React.useState<'typescript' | 'python'>('typescript');
  const [highlightedCode, setHighlightedCode] = React.useState<string>('');
  const highlighterRef = React.useRef<Highlighter | null>(null);

  React.useEffect(() => {
    async function initHighlighter() {
      try {
        highlighterRef.current = await getHighlighter({
          themes: ['github-dark', 'github-light'],
          langs: ['typescript', 'python'],
        });
        updateHighlightedCode();
      } catch (error) {
        console.error('Failed to initialize highlighter:', error);
      }
    }

    if (!highlighterRef.current) {
      initHighlighter();
    }
  }, []);

  const updateHighlightedCode = React.useCallback(() => {
    if (!highlighterRef.current) return;

    const code = activeTab === 'typescript' ? example.typescript : example.python;
    const language = activeTab === 'typescript' ? 'typescript' : 'python';
    const isDark = document.documentElement.classList.contains('dark');

    try {
      const highlighted = highlighterRef.current.codeToHtml(code, {
        lang: language,
        theme: isDark ? 'github-dark' : 'github-light',
      });
      setHighlightedCode(highlighted);
    } catch (error) {
      console.error('Failed to highlight code:', error);
      setHighlightedCode(`<pre><code>${code}</code></pre>`);
    }
  }, [activeTab, example.typescript, example.python]);

  React.useEffect(() => {
    updateHighlightedCode();
  }, [updateHighlightedCode]);

  // Listen for theme changes
  React.useEffect(() => {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (
          mutation.type === 'attributes' &&
          mutation.attributeName === 'class'
        ) {
          updateHighlightedCode();
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => observer.disconnect();
  }, [updateHighlightedCode]);

  return (
    <div className="terminal-card mt-6">
      <div className="flex items-center px-4 py-2 bg-editor-highlight dark:bg-editor-highlight-dark border-b border-editor-highlight dark:border-editor-highlight-dark">
        <div className="flex space-x-2">
          <button
            className={`px-3 py-1 rounded text-sm font-mono ${
              activeTab === 'typescript'
                ? 'bg-editor-accent bg-opacity-20 text-editor-accent'
                : 'text-editor-muted dark:text-editor-muted-dark hover:text-editor-text dark:hover:text-editor-text-dark'
            }`}
            onClick={() => setActiveTab('typescript')}
          >
            typescript
          </button>
          <button
            className={`px-3 py-1 rounded text-sm font-mono ${
              activeTab === 'python'
                ? 'bg-editor-accent bg-opacity-20 text-editor-accent'
                : 'text-editor-muted dark:text-editor-muted-dark hover:text-editor-text dark:hover:text-editor-text-dark'
            }`}
            onClick={() => setActiveTab('python')}
          >
            python
          </button>
        </div>
      </div>
      <div 
        className="p-4 bg-editor-bg dark:bg-editor-bg-dark font-mono text-sm overflow-x-auto"
        dangerouslySetInnerHTML={{ __html: highlightedCode }}
      />
      <div className="p-4 border-t border-editor-highlight dark:border-editor-highlight-dark">
        <p className="text-sm text-editor-muted dark:text-editor-muted-dark font-mono"># {example.description}</p>
      </div>
    </div>
  );
}