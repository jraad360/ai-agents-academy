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
        content: `AI agents are systems that use language models to decide the control flow of an application, enabling them to perceive their environment, make decisions, and take actions to achieve specific goals. Unlike basic LLMs that simply generate text, agents can interact with external tools and systems to accomplish complex tasks.

Key characteristics of AI agents include:
• Autonomy - They can operate independently with minimal supervision
• Reactivity - They respond to changes in their environment and adapt accordingly
• Goal-oriented - They work towards specific objectives through planning and execution
• Tool usage - They can leverage external functions to extend their capabilities
• Memory - They maintain context across interactions and reasoning steps

Learn more about agent concepts in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/).

Agent Components:
1. Language Model (LLM/Chat Model)
   - The reasoning engine of the agent
   - Makes decisions based on context and goals
   - Examples: GPT-4, Claude, Llama, Mistral, etc.

2. Tools
   - Functions the agent can use to interact with external systems
   - Examples: web search, calculators, APIs, databases
   - Custom tools for domain-specific tasks

3. Memory
   - Stores conversation history and intermediate results
   - Maintains context between interactions
   - Can be short-term (within a session) or long-term (across sessions)

4. Orchestration
   - Coordinates the flow between components
   - Manages the planning and execution loop
   - Handles error cases and recovery strategies`
      },
      {
        title: 'Agent Architectures',
        content: `Modern agent architectures follow different patterns depending on the complexity of tasks they need to handle. The most common architectures include:

1. Router Agents
   - Make a single decision from predefined options
   - Useful for classification or simple routing tasks
   - Limited control but efficient for specific use cases

2. Tool-Calling Agents (ReAct)
   - Combine reasoning and acting in an iterative process
   - Can use multiple tools in sequence to solve complex problems
   - Maintain memory of previous steps and observations
   - Most common general-purpose agent architecture

3. Planning Agents
   - Create explicit plans before execution
   - Break down complex tasks into manageable steps
   - Can revise plans based on new information
   - Useful for multi-step, complex reasoning tasks

4. Multi-Agent Systems
   - Multiple specialized agents working together
   - Different roles and responsibilities
   - Communication protocols between agents
   - Collaborative problem-solving

Learn more about agent types in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#agent-architectures).

The ReAct (Reasoning + Acting) pattern is particularly important as it forms the foundation of many modern agent implementations. It follows these steps:
1. Thought - The agent reasons about the current state and goal
2. Action - The agent selects and uses a tool
3. Observation - The agent processes the tool's output
4. Repeat until the goal is achieved`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Define a custom tool with input validation using Zod
const weatherTool = tool(
  async ({ location }) => {
    // In a real implementation, this would call a weather API
    if (location.toLowerCase().includes("san francisco")) {
      return "60°F and foggy";
    }
    return "75°F and sunny";
  },
  {
    name: "get_weather",
    description: "Get the current weather for a location",
    schema: z.object({
      location: z.string().describe("The city and state, e.g. San Francisco, CA"),
    }),
  }
);

// Create a calculator tool
const calculatorTool = tool(
  async ({ expression }) => {
    // Simple calculator that evaluates mathematical expressions
    try {
      return eval(expression).toString();
    } catch (error) {
      return "Error: Could not evaluate the expression";
    }
  },
  {
    name: "calculator",
    description: "Evaluate a mathematical expression",
    schema: z.object({
      expression: z.string().describe("The mathematical expression to evaluate"),
    }),
  }
);

// Initialize the language model
const model = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

// Create the ReAct agent
const agent = createReactAgent({
  llm: model,
  tools: [weatherTool, calculatorTool],
});

// Run the agent
const result = await agent.invoke({
  messages: [{
    role: "user",
    content: "What's the weather in San Francisco? Also, what's 24 * 7?"
  }]
});

console.log(result.messages[result.messages.length - 1].content);`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import TypedDict, Annotated

# Define a custom tool for weather information
@tool
def get_weather(location: Annotated[str, "The city and state, e.g. San Francisco, CA"]) -> str:
    """Get the current weather for a location."""
    # In a real implementation, this would call a weather API
    if "san francisco" in location.lower():
        return "60°F and foggy"
    return "75°F and sunny"

# Create a calculator tool
@tool
def calculator(expression: Annotated[str, "The mathematical expression to evaluate"]) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Create the ReAct agent
agent = create_react_agent(
    llm=model,
    tools=[get_weather, calculator],
)

# Run the agent
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What's the weather in San Francisco? Also, what's 24 * 7?"
    }]
})

print(result["messages"][-1].content)`,
          description: 'Creating a basic ReAct agent with custom tools'
        }]
      },
      {
        title: 'LangChain and LangGraph Ecosystem',
        content: `The LangChain ecosystem provides a comprehensive framework for building AI agents, with several key components:

1. LangChain Core (@langchain/core)
   - Base abstractions and interfaces
   - LangChain Expression Language (LCEL)
   - Core components like models, tools, and memory

2. LangChain (@langchain/langchain)
   - Higher-level components and chains
   - Pre-built agent implementations
   - Integration utilities

3. LangGraph (@langchain/langgraph)
   - Graph-based orchestration for agents
   - State management and persistence
   - Advanced control flow patterns
   - Multi-agent systems

4. LangSmith
   - Debugging and monitoring tools
   - Performance evaluation
   - Tracing and observability

The relationship between these components is hierarchical:
• LangChain Core provides the fundamental building blocks
• LangChain offers pre-built components and utilities
• LangGraph enables complex agent orchestration
• LangSmith helps with debugging and monitoring

Learn more about the ecosystem architecture in the [LangChain documentation](https://python.langchain.com/docs/concepts/architecture/).

When building agents, you'll typically:
1. Start with LangChain Core for basic components
2. Use LangChain for pre-built utilities and patterns
3. Leverage LangGraph for complex orchestration
4. Monitor and debug with LangSmith`,
        codeExamples: [{
          typescript: `// LangChain Core - Basic building blocks
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// LangGraph - Agent orchestration
import { StateGraph, END } from "@langchain/langgraph";
import { RunnableSequence } from "@langchain/core/runnables";

// Define a simple prompt
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant."],
  ["human", "{input}"]
]);

// Create a language model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0
});

// Create a simple chain with LCEL
const chain = prompt.pipe(model).pipe(new StringOutputParser());

// Define a state interface for our graph
interface GraphState {
  input: string;
  response?: string;
}

// Create a graph for more complex orchestration
const workflow = new StateGraph<GraphState>({
  channels: {
    input: {
      value: (x: string) => x,
      default: () => ""
    },
    response: {
      value: (x: string) => x,
      default: () => undefined
    }
  }
});

// Add a node to the graph
workflow.addNode("generate_response", async (state) => {
  const response = await chain.invoke({ input: state.input });
  return { response };
});

// Define the edges
workflow.setEntryPoint("generate_response");
workflow.addEdge("generate_response", END);

// Compile the graph
const app = workflow.compile();

// Run the graph
const result = await app.invoke({ input: "What is an AI agent?" });
console.log(result.response);`,
          python: `# LangChain Core - Basic building blocks
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph - Agent orchestration
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# Define a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# Create a language model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Create a simple chain with LCEL
chain = prompt | model | StrOutputParser()

# Define a state type for our graph
class GraphState(TypedDict):
    input: str
    response: Optional[str]

# Create a graph for more complex orchestration
workflow = StateGraph(GraphState)

# Define a node function
async def generate_response(state: GraphState) -> GraphState:
    response = await chain.ainvoke({"input": state["input"]})
    return {"response": response}

# Add a node to the graph
workflow.add_node("generate_response", generate_response)

# Define the edges
workflow.set_entry_point("generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()

# Run the graph
result = app.invoke({"input": "What is an AI agent?"})
print(result["response"])`,
          description: 'Using components from the LangChain ecosystem'
        }]
      }
    ]
  }
};
