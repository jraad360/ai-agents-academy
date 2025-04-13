import { Module } from '../../../types';

export const advancedPatternsModule: Module = {
  id: 'advanced-patterns',
  title: 'Advanced Patterns',
  description: 'Learn production-ready implementation patterns',
  duration: '2.5 hours',
  level: 'advanced',
  content: {
    sections: [
      {
        title: 'Multi-Agent Systems',
        content: `Multi-agent systems enable complex problem-solving through agent collaboration. Learn more about orchestration in the [LangGraph documentation](https://python.langchain.com/docs/langgraph).

Key Patterns:

1. Manager-Worker
   - Central coordinator
   - Task distribution
   - Result aggregation

2. Peer-to-Peer
   - Direct communication
   - Shared knowledge
   - Collaborative solving

3. Hierarchical
   - Specialized roles
   - Clear reporting lines
   - Escalation paths

Implementation Considerations:
• State management
• Communication protocols
• Error propagation
• Resource allocation`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent, AgentExecutor } from "langchain/agents";
import { StateGraph, END } from "langgraph";
import { RunnableSequence } from "@langchain/core/runnables";
import { pull } from "langchain/hub";

interface AgentState {
  messages: string[];
  next_agent: string;
}

async function createMultiAgentSystem() {
  // Create specialized agents
  const researchAgent = await createReactAgent({
    llm: new ChatOpenAI({ temperature: 0 }),
    tools: [new WebBrowser()],
    prompt: await pull("hwchase17/react-research")
  });

  const writingAgent = await createReactAgent({
    llm: new ChatOpenAI({ temperature: 0.7 }),
    tools: [],
    prompt: await pull("hwchase17/react-writer")
  });

  const editingAgent = await createReactAgent({
    llm: new ChatOpenAI({ temperature: 0.2 }),
    tools: [],
    prompt: await pull("hwchase17/react-editor")
  });

  // Create the workflow graph
  const workflow = new StateGraph<AgentState>({
    channels: ["messages", "next_agent"]
  });

  // Add nodes
  workflow.addNode("research", researchAgent);
  workflow.addNode("write", writingAgent);
  workflow.addNode("edit", editingAgent);

  // Define transitions
  workflow.addEdge("research", "write");
  workflow.addEdge("write", "edit");
  workflow.addEdge("edit", END);

  // Compile the workflow
  const chain = workflow.compile();

  return chain;
}

// Use the multi-agent system
const chain = await createMultiAgentSystem();
const result = await chain.invoke({
  input: "Create a comprehensive article about quantum computing",
  messages: [],
  next_agent: "research"
});`,
          python: `from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langgraph.graph import StateGraph, END
from langchain.hub import pull
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]
    next_agent: str

def create_multi_agent_system():
    # Create specialized agents
    research_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0),
        tools=[WebBrowser()],
        prompt=pull("hwchase17/react-research")
    )

    writing_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0.7),
        tools=[],
        prompt=pull("hwchase17/react-writer")
    )

    editing_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0.2),
        tools=[],
        prompt=pull("hwchase17/react-editor")
    )

    # Create the workflow graph
    workflow = StateGraph(channels=["messages", "next_agent"])

    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("write", writing_agent)
    workflow.add_node("edit", editing_agent)

    # Define transitions
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "edit")
    workflow.add_edge("edit", END)

    # Compile the workflow
    chain = workflow.compile()

    return chain

# Use the multi-agent system
chain = create_multi_agent_system()
result = chain.invoke({
    "input": "Create a comprehensive article about quantum computing",
    "messages": [],
    "next_agent": "research"
})`,
          description: 'Implementing a multi-agent system using LangGraph for complex workflows'
        }]
      }
    ]
  }
};