import { Section } from '../../types';
import { introModule } from './foundations/intro-new';
import { coreConceptsModule } from './foundations/core-concepts';
import { buildingBlocksModule } from './foundations/building-blocks';
import { agentArchitecturesModule } from './foundations/agent-architectures';
import { functionCallingAgentsModule } from './foundations/function-calling-agents';
import { langgraphFundamentalsModule } from './foundations/langgraph-fundamentals';
import { multiAgentSystemsModule } from './foundations/multi-agent-systems';

import { introModule as adkIntroModule } from './google-adk/intro';
import { buildingAgentsModule } from './google-adk/building-agents';
import { advancedFeaturesModule } from './google-adk/advanced-features';

export const sections: Section[] = [
  {
    id: 'langchain',
    title: 'LangChain & LangGraph',
    description: 'Master AI agent development with LangChain and LangGraph frameworks',
    modules: [
      introModule,
      coreConceptsModule,
      buildingBlocksModule,
      agentArchitecturesModule,
      functionCallingAgentsModule,
      langgraphFundamentalsModule,
      multiAgentSystemsModule,
    ]
  },
  {
    id: 'google-adk',
    title: 'Google Agent Developer Kit',
    description: 'Learn to build AI agents using Google\'s Agent Developer Kit',
    modules: [
      adkIntroModule,
      buildingAgentsModule,
      advancedFeaturesModule
    ]
  }
];