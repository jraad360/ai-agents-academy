export interface Section {
  id: string;
  title: string;
  description: string;
  modules: Module[];
}

export interface Module {
  id: string;
  title: string;
  description: string;
  duration: string;
  level: 'beginner' | 'intermediate' | 'advanced';
  content?: ModuleContent;
}

export interface ModuleContent {
  sections: ContentSection[];
}

export interface ContentSection {
  title: string;
  content: string;
  codeExamples?: CodeExample[];
}

export interface CodeExample {
  typescript: string;
  python: string;
  description: string;
}

export interface Theme {
  isDark: boolean;
  toggleTheme: () => void;
}