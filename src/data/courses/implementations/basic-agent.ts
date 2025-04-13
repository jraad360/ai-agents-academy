import { Module } from '../../../types';

export const basicAgentModule: Module = {
  id: 'basic-agent',
  title: 'Creating Your First Agent',
  description: 'Step-by-step guide to building a basic AI agent',
  duration: '2 hours',
  level: 'beginner',
  content: {
    sections: [
      {
        title: 'Setting Up Your First Agent',
        content: `Let's create a research agent that can search the web and summarize information. This implementation follows the latest [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/interface) patterns.

Key Features:
1. Web search capability
2. Content summarization
3. Citation tracking
4. Error handling

Implementation Steps:

1. Environment Setup
   - Install dependencies
   - Configure API keys
   - Set up error tracking

2. Tool Configuration
   - Web search setup
   - Content extraction
   - Summary generation

3. Agent Creation
   - Prompt engineering
   - Tool integration
   - Memory configuration

4. Testing & Validation
   - Input validation
   - Output verification
   - Error scenarios`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent, AgentExecutor } from "langchain/agents";
import { 
  WebBrowser,
  DuckDuckGoSearchResults
} from "@langchain/community/tools";
import { BufferMemory } from "langchain/memory";
import { pull } from "langchain/hub";

async function createResearchAgent() {
  // Initialize the model
  const model = new ChatOpenAI({
    modelName: "gpt-4",
    temperature: 0.2,
    streaming: true
  });

  // Configure tools with error handling
  const tools = [
    new WebBrowser({
      maxPages: 3,
      timeout: 5000
    }),
    new DuckDuckGoSearchResults({
      maxResults: 5
    })
  ];

  // Create memory system
  const memory = new BufferMemory({
    returnMessages: true,
    memoryKey: "chat_history",
    inputKey: "input",
    outputKey: "output"
  });

  // Initialize the agent
  const agent = await createReactAgent({
    llm: model,
    tools,
    prompt: await pull("hwchase17/react-research")
  });

  // Create executor with configuration
  return new AgentExecutor({
    agent,
    tools,
    memory,
    verbose: true,
    maxIterations: 5,
    returnIntermediateSteps: true,
    earlyStoppingMethod: "force",
    handleParsingErrors: true
  });
}

// Use the agent with error handling
try {
  const agent = await createResearchAgent();
  const result = await agent.invoke({
    input: "Research the latest developments in quantum computing"
  });

  console.log("Final Answer:", result.output);
  console.log("Sources:", result.intermediateSteps
    .filter(step => step.action.tool === "web_browser")
    .map(step => step.action.toolInput.url)
  );
} catch (error) {
  console.error("Agent execution failed:", error);
}`,
          python: `from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import WebBrowser, DuckDuckGoSearchResults
from langchain.memory import BufferMemory
from langchain.hub import pull
from typing import List, Dict
import logging

def create_research_agent():
    # Initialize the model
    model = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.2,
        streaming=True
    )

    # Configure tools with error handling
    tools = [
        WebBrowser(
            max_pages=3,
            timeout=5000
        ),
        DuckDuckGoSearchResults(
            max_results=5
        )
    ]

    # Create memory system
    memory = BufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="input",
        output_key="output"
    )

    # Initialize the agent
    agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt=pull("hwchase17/react-research")
    )

    # Create executor with configuration
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
        return_intermediate_steps=True,
        early_stopping_method="force",
        handle_parsing_errors=True
    )

# Use the agent with error handling
try:
    agent = create_research_agent()
    result = agent.invoke({
        "input": "Research the latest developments in quantum computing"
    })

    print("Final Answer:", result["output"])
    print("Sources:", [
        step.action.tool_input["url"]
        for step in result["intermediate_steps"]
        if step.action.tool == "web_browser"
    ])
except Exception as e:
    logging.error(f"Agent execution failed: {str(e)}")`,
          description: 'Creating a research agent with web search and summarization capabilities'
        }]
      }
    ]
  }
};