import { Module } from '../../../types';

export const agentArchitecturesModule: Module = {
  id: 'agent-architectures',
  title: 'Agent Architectures',
  description: 'Understanding different agent design patterns and implementation strategies',
  duration: '2 hours',
  level: 'intermediate',
  content: {
    sections: [
      {
        title: 'ReAct Pattern',
        content: `The ReAct (Reasoning + Acting) pattern is a fundamental agent architecture that combines reasoning and action in an iterative cycle. It's one of the most widely used patterns for building general-purpose AI agents.

Key concepts of the ReAct pattern:

• Thought-Action-Observation Cycle
  - Thought: The agent reasons about the current state and goal
  - Action: The agent selects and uses a tool
  - Observation: The agent processes the tool's output
  - Repeat until the goal is achieved

• Benefits
  - Enables complex multi-step reasoning
  - Provides transparency into the agent's decision process
  - Allows for course correction based on observations
  - Works well with a wide range of tasks

• Implementation Approaches
  - Native function calling with modern LLMs
  - Structured output parsing for older models
  - Hybrid approaches for specialized needs

The ReAct pattern has evolved significantly since its introduction. Modern implementations leverage native function calling capabilities of LLMs rather than relying on text parsing, making agents more reliable and easier to build.

Learn more about the ReAct pattern in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-implementation).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";

// Define tools for the agent
const searchTool = tool(
  async ({ query }) => {
    // In a real implementation, this would call a search API
    if (query.toLowerCase().includes("weather")) {
      return "The weather is currently sunny with a high of 75°F.";
    } else if (query.toLowerCase().includes("news")) {
      return "Latest headlines: New AI breakthrough announced today.";
    }
    return "No relevant results found.";
  },
  {
    name: "search",
    description: "Search the web for information",
    schema: z.object({
      query: z.string().describe("The search query"),
    }),
  }
);

const calculatorTool = tool(
  async ({ expression }) => {
    try {
      // Using Function constructor for safe evaluation
      return new Function('return ' + expression)().toString();
    } catch (error) {
      return 'Error calculating result: ' + error.message;
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
  tools: [searchTool, calculatorTool],
});

// Run the agent
async function runAgent() {
  const result = await agent.invoke({
    messages: [
      new HumanMessage("What's the weather like today? Also, calculate 24 * 7.")
    ]
  });

  // Print the final response
  console.log(result.messages[result.messages.length - 1].content);

  // Print the intermediate steps (for debugging)
  for (const message of result.messages) {
    console.log(message.type + ': ' + JSON.stringify(message.content));
  }
}

runAgent();`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage

# Define tools for the agent
@tool
def search(query: Annotated[str, "The search query"]) -> str:
    """Search the web for information."""
    # In a real implementation, this would call a search API
    if "weather" in query.lower():
        return "The weather is currently sunny with a high of 75°F."
    elif "news" in query.lower():
        return "Latest headlines: New AI breakthrough announced today."
    return "No relevant results found."

@tool
def calculator(expression: Annotated[str, "The mathematical expression to evaluate"]) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating result: {str(e)}"

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Create the ReAct agent
agent = create_react_agent(
    llm=model,
    tools=[search, calculator],
)

# Run the agent
def run_agent():
    result = agent.invoke({
        "messages": [
            HumanMessage(content="What's the weather like today? Also, calculate 24 * 7.")
        ]
    })

    # Print the final response
    print(result["messages"][-1].content)

    # Print the intermediate steps (for debugging)
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")

run_agent()`,
          description: 'Creating a basic ReAct agent with search and calculator tools'
        }]
      },
      {
        title: 'Planning Agents',
        content: `Planning agents are designed to handle complex tasks by first creating a plan and then executing it step by step. This architecture is particularly effective for tasks that require multiple steps, dependencies, or careful sequencing.

Key concepts of planning agents:

• Plan-and-Execute Pattern
  - Planning Phase: Create a detailed plan with specific steps
  - Execution Phase: Carry out each step, potentially adapting the plan
  - Verification: Ensure the goal has been achieved

• Types of Planning Agents
  - Simple Planners: Create a linear sequence of steps
  - Hierarchical Planners: Break down complex tasks into subtasks
  - Dynamic Planners: Revise plans based on new information
  - Recursive Planners: Apply planning recursively to subtasks

• Benefits
  - Better handling of complex, multi-step tasks
  - Improved efficiency through proper sequencing
  - Enhanced transparency and explainability
  - Reduced likelihood of missing critical steps

Planning agents are particularly useful for tasks like research, content creation, data analysis, and problem-solving where a systematic approach is beneficial.

Learn more about planning agents in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#planning).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";
import { RunnableSequence } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";

// Define the state interface
interface PlannerState {
  input: string;
  plan?: string[];
  currentStepIndex?: number;
  stepResults?: Record<number, string>;
  output?: string;
}

// Initialize the language model
const model = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

// Create a planner prompt
const plannerPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a planning assistant that creates step-by-step plans to accomplish tasks. Break down the task into 3-5 clear, sequential steps."],
  ["human", "Task: {input}"]
]);

// Create a planner chain
const planner = RunnableSequence.from([
  plannerPrompt,
  model,
  new StringOutputParser(),
  (text: string) => {
    // Extract steps from the model's response
    const steps = text
      .split("\\n")
      .filter(line => /^\\d+\./.test(line))
      .map(line => line.replace(/^\\d+\.\s*/, "").trim());
    return steps;
  }
]);

// Define tools for execution
const searchTool = tool(
  async ({ query }) => {
    // In a real implementation, this would call a search API
    return 'Search results for "' + query + '": Found relevant information about ' + query + '.';
  },
  {
    name: "search",
    description: "Search for information",
    schema: z.object({
      query: z.string().describe("The search query"),
    }),
  }
);

const summarizeTool = tool(
  async ({ text }) => {
    // In a real implementation, this would use an LLM to summarize
    return 'Summary of "' + text.substring(0, 30) + '...": This is a concise summary.';
  },
  {
    name: "summarize",
    description: "Summarize text",
    schema: z.object({
      text: z.string().describe("The text to summarize"),
    }),
  }
);

// Create an executor prompt
const executorPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a task execution assistant. Execute the current step using the available tools if needed."],
  ["human", "\nTask: {input}\nPlan: {plan}\nCurrent Step ({currentStepIndex}): {currentStep}\n\nPrevious step results:\n{previousResults}\n\nExecute this step. Use tools if necessary.\n"]
]);

// Create an executor chain
const executor = RunnableSequence.from([
  executorPrompt,
  model.bind({
    tools: [searchTool, summarizeTool]
  }),
  new StringOutputParser()
]);

// Create a finalizer prompt
const finalizerPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a task completion assistant. Provide a final response based on all the steps executed."],
  ["human", "\nTask: {input}\nPlan: {plan}\nStep Results:\n{allResults}\n\nProvide a comprehensive final response that addresses the original task.\n"]
]);

// Create a finalizer chain
const finalizer = RunnableSequence.from([
  finalizerPrompt,
  model,
  new StringOutputParser()
]);

// Create the planning agent graph
const workflow = new StateGraph<PlannerState>({
  channels: {
    input: {
      value: (x: string) => x,
      default: () => ""
    },
    plan: {
      value: (x: string[]) => x,
      default: () => undefined
    },
    currentStepIndex: {
      value: (x: number) => x,
      default: () => undefined
    },
    stepResults: {
      value: (x: Record<number, string>) => x,
      default: () => ({})
    },
    output: {
      value: (x: string) => x,
      default: () => undefined
    }
  }
});

// Add nodes to the graph
workflow.addNode("create_plan", async (state) => {
  const plan = await planner.invoke({ input: state.input });
  return {
    plan,
    currentStepIndex: 0,
    stepResults: {}
  };
});

workflow.addNode("execute_step", async (state) => {
  const currentStep = state.plan[state.currentStepIndex];

  // Format previous results for context
  let previousResults = "";
  for (let i = 0; i < state.currentStepIndex; i++) {
    previousResults += 'Step ' + (i + 1) + ': ' + state.stepResults[i] + '\n';
  }

  // Execute the current step
  const result = await executor.invoke({
    input: state.input,
    plan: state.plan.join("\\n"),
    currentStepIndex: state.currentStepIndex + 1,
    currentStep,
    previousResults
  });

  // Update step results
  const updatedResults = { ...state.stepResults };
  updatedResults[state.currentStepIndex] = result;

  // Check if we've completed all steps
  const nextIndex = state.currentStepIndex + 1;
  if (nextIndex >= state.plan.length) {
    return {
      stepResults: updatedResults,
      currentStepIndex: nextIndex
    };
  }

  // Move to the next step
  return {
    stepResults: updatedResults,
    currentStepIndex: nextIndex
  };
});

workflow.addNode("finalize", async (state) => {
  // Format all results for the finalizer
  let allResults = "";
  for (let i = 0; i < state.plan.length; i++) {
    allResults += 'Step ' + (i + 1) + ' (' + state.plan[i] + '): ' + state.stepResults[i] + '\n';
  }

  // Generate the final output
  const output = await finalizer.invoke({
    input: state.input,
    plan: state.plan.join("\\n"),
    allResults
  });

  return { output };
});

// Define the edges
workflow.setEntryPoint("create_plan");
workflow.addEdge("create_plan", "execute_step");

// Conditional edge: if we have more steps, continue executing
workflow.addConditionalEdges(
  "execute_step",
  (state) => {
    if (state.currentStepIndex < state.plan.length) {
      return "execute_step";
    }
    return "finalize";
  }
);

workflow.addEdge("finalize", END);

// Compile the graph
const planningAgent = workflow.compile();

// Run the planning agent
async function runPlanningAgent() {
  const result = await planningAgent.invoke({
    input: "Research the latest advancements in quantum computing and prepare a summary."
  });

  console.log("Final Plan:", result.plan);
  console.log("Step Results:", result.stepResults);
  console.log("Final Output:", result.output);
}

runPlanningAgent();`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableSequence
from typing import TypedDict, List, Dict, Optional, Annotated

# Define the state type
class PlannerState(TypedDict):
    input: str
    plan: Optional[List[str]]
    current_step_index: Optional[int]
    step_results: Dict[int, str]
    output: Optional[str]

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Create a planner prompt
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a planning assistant that creates step-by-step plans to accomplish tasks. Break down the task into 3-5 clear, sequential steps."),
    ("human", "Task: {input}")
])

# Create a planner chain
def extract_steps(text: str) -> List[str]:
    # Extract steps from the model's response
    steps = []
    for line in text.split("\\n"):
        if line.strip() and line.strip()[0].isdigit() and "." in line:
            steps.append(line.split(".", 1)[1].strip())
    return steps

planner = (
    planner_prompt
    | model
    | StrOutputParser()
    | extract_steps
)

# Define tools for execution
@tool
def search(query: Annotated[str, "The search query"]) -> str:
    """Search for information."""
    # In a real implementation, this would call a search API
    return 'Search results for "' + query + '": Found relevant information about ' + query + '.'

@tool
def summarize(text: Annotated[str, "The text to summarize"]) -> str:
    """Summarize text."""
    # In a real implementation, this would use an LLM to summarize
    return 'Summary of "' + text[:30] + '...": This is a concise summary.'

# Create an executor prompt
executor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a task execution assistant. Execute the current step using the available tools if needed."),
    ("human", """
Task: {input}
Plan: {plan}
Current Step ({current_step_index}): {current_step}

Previous step results:
{previous_results}

Execute this step. Use tools if necessary.
""")
])

# Create an executor chain
executor = (
    executor_prompt
    | model.bind_tools([search, summarize])
    | StrOutputParser()
)

# Create a finalizer prompt
finalizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a task completion assistant. Provide a final response based on all the steps executed."),
    ("human", """
Task: {input}
Plan: {plan}
Step Results:
{all_results}

Provide a comprehensive final response that addresses the original task.
""")
])

# Create a finalizer chain
finalizer = (
    finalizer_prompt
    | model
    | StrOutputParser()
)

# Create the planning agent graph
workflow = StateGraph(PlannerState)

# Add nodes to the graph
def create_plan(state: PlannerState) -> PlannerState:
    plan = planner.invoke({"input": state["input"]})
    return {
        "plan": plan,
        "current_step_index": 0,
        "step_results": {}
    }

async def execute_step(state: PlannerState) -> PlannerState:
    current_step = state["plan"][state["current_step_index"]]

    # Format previous results for context
    previous_results = ""
    for i in range(state["current_step_index"]):
        previous_results += "Step " + str(i + 1) + ": " + state['step_results'][i] + "\n"

    # Execute the current step
    result = await executor.ainvoke({
        "input": state["input"],
        "plan": "\\n".join(state["plan"]),
        "current_step_index": state["current_step_index"] + 1,
        "current_step": current_step,
        "previous_results": previous_results
    })

    # Update step results
    updated_results = state["step_results"].copy()
    updated_results[state["current_step_index"]] = result

    # Move to the next step
    next_index = state["current_step_index"] + 1

    return {
        "step_results": updated_results,
        "current_step_index": next_index
    }

async def finalize(state: PlannerState) -> PlannerState:
    # Format all results for the finalizer
    all_results = ""
    for i in range(len(state["plan"])):
        all_results += "Step " + str(i + 1) + " (" + state['plan'][i] + "): " + state['step_results'][i] + "\n"

    # Generate the final output
    output = await finalizer.ainvoke({
        "input": state["input"],
        "plan": "\\n".join(state["plan"]),
        "all_results": all_results
    })

    return {"output": output}

workflow.add_node("create_plan", create_plan)
workflow.add_node("execute_step", execute_step)
workflow.add_node("finalize", finalize)

# Define the edges
workflow.set_entry_point("create_plan")
workflow.add_edge("create_plan", "execute_step")

# Conditional edge: if we have more steps, continue executing
def next_step_or_finalize(state: PlannerState) -> str:
    if state["current_step_index"] < len(state["plan"]):
        return "execute_step"
    return "finalize"

workflow.add_conditional_edges(
    "execute_step",
    next_step_or_finalize
)

workflow.add_edge("finalize", END)

# Compile the graph
planning_agent = workflow.compile()

# Run the planning agent
async def run_planning_agent():
    result = await planning_agent.ainvoke({
        "input": "Research the latest advancements in quantum computing and prepare a summary."
    })

    print("Final Plan:", result["plan"])
    print("Step Results:", result["step_results"])
    print("Final Output:", result["output"])

import asyncio
asyncio.run(run_planning_agent())`,
          description: 'Implementing a plan-and-execute agent with planning, execution, and finalization phases'
        }]
      }
    ]
  }
};
