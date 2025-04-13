import { Module } from '../../../types';

export const introModule: Module = {
  id: 'intro',
  title: 'Introduction to AI Agents',
  description: 'Learn what AI agents are and how they work',
  duration: '45 mins',
  level: 'beginner',
  content: {
    sections: [
      {
        title: 'What are AI Agents?',
        content: `AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve specific goals. In the context of LangChain, agents are powerful abstractions that combine language models with tools and memory to solve complex tasks.

Key characteristics of AI agents include:
• Autonomy - They can operate independently
• Reactivity - They respond to changes in their environment
• Goal-oriented - They work towards specific objectives
• Flexibility - They can handle various types of tasks

Learn more about the core concepts in the [LangChain documentation](https://python.langchain.com/docs/modules/agents).

Agent Components:
1. Language Model (LLM/Chat Model)
   - The reasoning engine of the agent
   - Makes decisions based on context and goals
   - Can be GPT-4, Claude, or other models

2. Tools
   - Functions the agent can use
   - Examples: web search, calculators, APIs
   - Custom tools for specific tasks

3. Memory
   - Stores conversation history
   - Maintains context between interactions
   - Different types for different needs

4. Prompt Templates
   - Guide agent behavior
   - Define interaction patterns
   - Set constraints and goals`
      },
      {
        title: 'Agent Architecture',
        content: `The modern agent architecture in LangChain follows a structured approach with several key components working together. This architecture is based on the latest [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) design.

Core Components:

1. Agent Executor
   - Orchestrates the agent's operation
   - Manages the planning and execution loop
   - Handles errors and retries

2. Agent
   - Contains the core logic
   - Implements specific agent patterns
   - Examples: ReAct, OpenAI Functions, Plan-and-Execute

3. Tools
   - Structured function interfaces
   - Built-in and custom implementations
   - Tool retrieval and selection

4. Memory Systems
   - Conversation buffers
   - Vector stores
   - Entity memory

Learn more about agent types in the [official documentation](https://python.langchain.com/docs/modules/agents/agent_types/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { AgentExecutor, createReactAgent } from "langchain/agents";
import { pull } from "langchain/hub";
import { Calculator, WebBrowser } from "@langchain/community/tools";

// Initialize the language model
const model = new ChatOpenAI({
  modelName: "gpt-4",
  temperature: 0
});

// Define available tools
const tools = [
  new Calculator(),
  new WebBrowser()
];

// Create the agent using LCEL
const agent = await createReactAgent({
  llm: model,
  tools,
  // Use standardized prompts from LangChain Hub
  prompt: await pull("hwchase17/react")
});

// Create the executor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
  verbose: true,
  // Add memory for context
  memory: new BufferMemory({
    memoryKey: "chat_history"
  })
});

// Run the agent
const result = await agentExecutor.invoke({
  input: "What is the population of France divided by 2?"
});`,
          python: `from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import Calculator, WebBrowser
from langchain.memory import BufferMemory
from langchain.hub import pull

# Initialize the language model
model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0
)

# Define available tools
tools = [
    Calculator(),
    WebBrowser()
]

# Create the agent using LCEL
agent = create_react_agent(
    llm=model,
    tools=tools,
    # Use standardized prompts from LangChain Hub
    prompt=pull("hwchase17/react")
)

# Create the executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    # Add memory for context
    memory=BufferMemory(
        memory_key="chat_history"
    )
)

# Run the agent
result = agent_executor.invoke({
    "input": "What is the population of France divided by 2?"
})`,
          description: 'Modern agent setup using LangChain Expression Language (LCEL)'
        }]
      }
    ]
  }
};