import { Module } from '../../../types';

export const functionCallingAgentsModule: Module = {
  id: 'function-calling-agents',
  title: 'Function Calling Agents',
  description: 'Understanding how to build agents that leverage native function calling capabilities',
  duration: '1.5 hours',
  level: 'intermediate',
  content: {
    sections: [
      {
        title: 'Native Function Calling',
        content: `Function calling agents leverage the native function calling capabilities of modern LLMs to interact with external tools and APIs. This architecture provides a more reliable and structured approach to tool usage compared to text parsing methods.

Key concepts of function calling agents:

• Native Function Calling
  - Models are specifically trained to recognize when to use tools
  - Output follows a structured JSON format for tool invocation
  - Reduces parsing errors and hallucinations in tool usage

• Benefits
  - More reliable tool selection and parameter extraction
  - Cleaner integration with external systems
  - Better handling of complex parameter types
  - Reduced prompt engineering requirements

• Implementation Approaches
  - Direct model function calling (OpenAI, Anthropic, etc.)
  - Tool binding with LangChain/LangGraph
  - Custom function calling with structured output parsing

Function calling agents are particularly well-suited for tasks that require precise interaction with external systems, such as database queries, API calls, and complex data transformations.

Learn more about function calling in the [LangChain documentation](https://python.langchain.com/docs/concepts/tools/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";

// Define a database query tool
const queryDatabaseTool = tool(
  async ({ table, filters, limit }) => {
    // In a real implementation, this would query a database
    console.log('Querying table: ' + table);
    console.log('Filters: ' + JSON.stringify(filters));
    console.log('Limit: ' + limit);

    return JSON.stringify({
      results: [
        { id: 1, name: "Product A", price: 29.99 },
        { id: 2, name: "Product B", price: 49.99 }
      ],
      count: 2
    });
  },
  {
    name: "query_database",
    description: "Query a database table with filters",
    schema: z.object({
      table: z.string().describe("The name of the table to query"),
      filters: z.record(z.any()).describe("Filters to apply to the query"),
      limit: z.number().optional().describe("Maximum number of results to return")
    }),
  }
);

// Define an API call tool
const callApiTool = tool(
  async ({ endpoint, method, params }) => {
    // In a real implementation, this would make an API request
    console.log('Calling API endpoint: ' + endpoint);
    console.log('Method: ' + method);
    console.log('Params: ' + JSON.stringify(params));

    return JSON.stringify({
      status: "success",
      data: {
        id: "12345",
        result: "Operation completed successfully"
      }
    });
  },
  {
    name: "call_api",
    description: "Make an API call to an external service",
    schema: z.object({
      endpoint: z.string().describe("The API endpoint to call"),
      method: z.enum(["GET", "POST", "PUT", "DELETE"]).describe("The HTTP method to use"),
      params: z.record(z.any()).describe("Parameters to include in the request")
    }),
  }
);

// Initialize the language model with function calling
const model = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
}).bind({
  tools: [queryDatabaseTool, callApiTool]
});

// Run the function calling agent
async function runFunctionCallingAgent() {
  const userMessage = new HumanMessage(
    "Find all products that cost less than $50 and then call the inventory API to check their stock levels."
  );

  const response = await model.invoke([userMessage]);
  console.log("Response:", response);
}

runFunctionCallingAgent();`,
          python: `from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Dict, Any, List, Optional, Annotated
from langchain_core.messages import HumanMessage

# Define a database query tool
@tool
def query_database(
    table: Annotated[str, "The name of the table to query"],
    filters: Annotated[Dict[str, Any], "Filters to apply to the query"],
    limit: Annotated[Optional[int], "Maximum number of results to return"] = None
) -> str:
    """Query a database table with filters."""
    # In a real implementation, this would query a database
    print(f"Querying table: {table}")
    print(f"Filters: {filters}")
    print(f"Limit: {limit}")

    return {
        "results": [
            {"id": 1, "name": "Product A", "price": 29.99},
            {"id": 2, "name": "Product B", "price": 49.99}
        ],
        "count": 2
    }

# Define an API call tool
@tool
def call_api(
    endpoint: Annotated[str, "The API endpoint to call"],
    method: Annotated[str, "The HTTP method to use (GET, POST, PUT, DELETE)"],
    params: Annotated[Dict[str, Any], "Parameters to include in the request"]
) -> str:
    """Make an API call to an external service."""
    # In a real implementation, this would make an API request
    print(f"Calling API endpoint: {endpoint}")
    print(f"Method: {method}")
    print(f"Params: {params}")

    return {
        "status": "success",
        "data": {
            "id": "12345",
            "result": "Operation completed successfully"
        }
    }

# Initialize the language model with function calling
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
).bind_tools([query_database, call_api])

# Run the function calling agent
def run_function_calling_agent():
    user_message = HumanMessage(
        content="Find all products that cost less than $50 and then call the inventory API to check their stock levels."
    )

    response = model.invoke([user_message])
    print("Response:", response)

run_function_calling_agent()`,
          description: 'Implementing a function calling agent with database and API tools'
        }]
      },
      {
        title: 'Autonomous Agents',
        content: `Autonomous agents are designed to operate with minimal human supervision, making independent decisions to achieve long-term goals. These agents combine multiple capabilities to create more versatile and self-directed systems.

Key concepts of autonomous agents:

• Core Capabilities
  - Long-term memory and context retention
  - Self-improvement and learning mechanisms
  - Goal setting and prioritization
  - Feedback incorporation and adaptation

• Architecture Components
  - Memory systems (short-term and long-term)
  - Planning and execution modules
  - Reflection and self-evaluation mechanisms
  - Safety guardrails and constraints

• Implementation Considerations
  - Balancing autonomy with safety
  - Managing computational resources
  - Handling failures and unexpected situations
  - Providing appropriate oversight mechanisms

Autonomous agents are particularly valuable for ongoing tasks that require persistence and adaptation, such as research assistants, personal productivity agents, and monitoring systems.

Learn more about autonomous agent architectures in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#custom-agent-architectures).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { RunnableSequence } from "@langchain/core/runnables";
import { HumanMessage } from "@langchain/core/messages";

// Define the agent's state
interface AgentState {
  messages: any[];
  memory: Record<string, any>;
  task_queue: string[];
  current_task?: string;
  task_results: Record<string, any>;
}

// Initialize the language model
const model = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

// Define tools
const searchTool = tool(
  async ({ query }) => {
    // In a real implementation, this would call a search API
    return 'Search results for "' + query + '"';
  },
  {
    name: "search",
    description: "Search for information",
    schema: z.object({
      query: z.string().describe("The search query"),
    }),
  }
);

// Create a task planning prompt
const taskPlanningPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are an autonomous agent that breaks down complex goals into specific tasks.\nGiven the current goal and memory, create a prioritized list of tasks to accomplish the goal.\nEach task should be specific and actionable."],
  ["human", "\nGoal: {goal}\nCurrent Memory: {memory}\n\nGenerate a list of tasks to accomplish this goal. Return only the tasks as a comma-separated list.\n"]
]);

// Create a task execution prompt
const taskExecutionPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are an autonomous agent that executes tasks to achieve goals.\nUse the available tools when necessary to complete the current task.\nBe thorough but concise in your actions."],
  ["human", "\nGoal: {goal}\nCurrent Task: {current_task}\nMemory: {memory}\n\nComplete this task. Use tools when necessary.\n"]
]);

// Create the agent graph
const workflow = new StateGraph<AgentState>({
  channels: {
    messages: {
      value: (x: any[]) => x,
      default: () => []
    },
    memory: {
      value: (x: Record<string, any>) => x,
      default: () => ({})
    },
    task_queue: {
      value: (x: string[]) => x,
      default: () => []
    },
    current_task: {
      value: (x: string) => x,
      default: () => undefined
    },
    task_results: {
      value: (x: Record<string, any>) => x,
      default: () => ({})
    }
  }
});

// Add nodes to the graph
workflow.addNode("plan_tasks", async (state) => {
  // Get the latest user message
  const lastMessage = state.messages[state.messages.length - 1];
  const goal = lastMessage.content;

  // Generate tasks
  const taskPlanningChain = RunnableSequence.from([
    taskPlanningPrompt,
    model,
    new StringOutputParser()
  ]);

  const tasksText = await taskPlanningChain.invoke({
    goal,
    memory: JSON.stringify(state.memory)
  });

  // Parse tasks into an array
  const tasks = tasksText
    .split(",")
    .map(task => task.trim())
    .filter(task => task.length > 0);

  return {
    task_queue: tasks,
    current_task: tasks[0],
    task_queue: tasks.slice(1)
  };
});

workflow.addNode("execute_task", async (state) => {
  if (!state.current_task) {
    return {};
  }

  // Get the latest user message
  const lastMessage = state.messages[state.messages.length - 1];
  const goal = lastMessage.content;

  // Execute the current task
  const taskExecutionChain = RunnableSequence.from([
    taskExecutionPrompt,
    model.bind({
      tools: [searchTool]
    }),
    new StringOutputParser()
  ]);

  const result = await taskExecutionChain.invoke({
    goal,
    current_task: state.current_task,
    memory: JSON.stringify(state.memory)
  });

  // Update task results
  const updatedResults = { ...state.task_results };
  updatedResults[state.current_task] = result;

  // Move to the next task if available
  const nextTask = state.task_queue.length > 0 ? state.task_queue[0] : undefined;
  const remainingTasks = state.task_queue.length > 0 ? state.task_queue.slice(1) : [];

  return {
    task_results: updatedResults,
    current_task: nextTask,
    task_queue: remainingTasks,
    // Update memory with the result
    memory: {
      ...state.memory,
      [state.current_task]: result
    }
  };
});

workflow.addNode("generate_response", async (state) => {
  // Generate a final response based on all task results
  const responsePrompt = ChatPromptTemplate.fromMessages([
    ["system", "You are an autonomous agent that provides comprehensive responses based on completed tasks."],
    ["human", "\nGoal: {goal}\nCompleted Tasks and Results:\n{task_results}\n\nProvide a comprehensive response that addresses the original goal.\n"]
  ]);

  const responseChain = RunnableSequence.from([
    responsePrompt,
    model,
    new StringOutputParser()
  ]);

  // Get the latest user message
  const lastMessage = state.messages[state.messages.length - 1];
  const goal = lastMessage.content;

  const response = await responseChain.invoke({
    goal,
    task_results: JSON.stringify(state.task_results, null, 2)
  });

  // Add the response to messages
  return {
    messages: [...state.messages, { role: "assistant", content: response }]
  };
});

// Define the edges
workflow.setEntryPoint("plan_tasks");
workflow.addEdge("plan_tasks", "execute_task");

// Conditional edge: if we have more tasks, continue executing
workflow.addConditionalEdges(
  "execute_task",
  (state) => {
    if (state.current_task) {
      return "execute_task";
    }
    return "generate_response";
  }
);

workflow.addEdge("generate_response", END);

// Compile the graph
const autonomousAgent = workflow.compile();

// Run the autonomous agent
async function runAutonomousAgent() {
  const result = await autonomousAgent.invoke({
    messages: [
      new HumanMessage("Research the latest advancements in renewable energy.")
    ]
  });

  console.log("Final Response:", result.messages[result.messages.length - 1].content);
}

runAutonomousAgent();`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import json

# Define the agent's state
class AgentState(TypedDict):
    messages: List[Any]
    memory: Dict[str, Any]
    task_queue: List[str]
    current_task: Optional[str]
    task_results: Dict[str, Any]

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Define tools
@tool
def search(query: Annotated[str, "The search query"]) -> str:
    """Search for information."""
    # In a real implementation, this would call a search API
    return 'Search results for "' + query + '"'

# Create a task planning prompt
task_planning_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an autonomous agent that breaks down complex goals into specific tasks.
Given the current goal and memory, create a prioritized list of tasks to accomplish the goal.
Each task should be specific and actionable."""),
    ("human", """
Goal: {goal}
Current Memory: {memory}

Generate a list of tasks to accomplish this goal. Return only the tasks as a comma-separated list.
""")
])

# Create a task execution prompt
task_execution_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an autonomous agent that executes tasks to achieve goals.
Use the available tools when necessary to complete the current task.
Be thorough but concise in your actions."""),
    ("human", """
Goal: {goal}
Current Task: {current_task}
Memory: {memory}

Complete this task. Use tools when necessary.
""")
])

# Create the agent graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
async def plan_tasks(state: AgentState) -> AgentState:
    # Get the latest user message
    last_message = state["messages"][-1]
    goal = last_message.content

    # Generate tasks
    task_planning_chain = (
        task_planning_prompt
        | model
        | StrOutputParser()
    )

    tasks_text = await task_planning_chain.ainvoke({
        "goal": goal,
        "memory": json.dumps(state["memory"])
    })

    # Parse tasks into a list
    tasks = [
        task.strip()
        for task in tasks_text.split(",")
        if task.strip()
    ]

    return {
        "task_queue": tasks[1:] if tasks else [],
        "current_task": tasks[0] if tasks else None
    }

async def execute_task(state: AgentState) -> AgentState:
    if not state.get("current_task"):
        return {}

    # Get the latest user message
    last_message = state["messages"][-1]
    goal = last_message.content

    # Execute the current task
    task_execution_chain = (
        task_execution_prompt
        | model.bind_tools([search])
        | StrOutputParser()
    )

    result = await task_execution_chain.ainvoke({
        "goal": goal,
        "current_task": state["current_task"],
        "memory": json.dumps(state["memory"])
    })

    # Update task results
    updated_results = state["task_results"].copy()
    updated_results[state["current_task"]] = result

    # Move to the next task if available
    next_task = state["task_queue"][0] if state["task_queue"] else None
    remaining_tasks = state["task_queue"][1:] if state["task_queue"] else []

    # Update memory with the result
    updated_memory = state["memory"].copy()
    updated_memory[state["current_task"]] = result

    return {
        "task_results": updated_results,
        "current_task": next_task,
        "task_queue": remaining_tasks,
        "memory": updated_memory
    }

async def generate_response(state: AgentState) -> AgentState:
    # Generate a final response based on all task results
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an autonomous agent that provides comprehensive responses based on completed tasks."),
        ("human", """
Goal: {goal}
Completed Tasks and Results:
{task_results}

Provide a comprehensive response that addresses the original goal.
""")
    ])

    response_chain = (
        response_prompt
        | model
        | StrOutputParser()
    )

    # Get the latest user message
    last_message = state["messages"][-1]
    goal = last_message.content

    response = await response_chain.ainvoke({
        "goal": goal,
        "task_results": json.dumps(state["task_results"], indent=2)
    })

    # Add the response to messages
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response}]
    }

workflow.add_node("plan_tasks", plan_tasks)
workflow.add_node("execute_task", execute_task)
workflow.add_node("generate_response", generate_response)

# Define the edges
workflow.set_entry_point("plan_tasks")
workflow.add_edge("plan_tasks", "execute_task")

# Conditional edge: if we have more tasks, continue executing
def next_step(state: AgentState) -> str:
    if state.get("current_task"):
        return "execute_task"
    return "generate_response"

workflow.add_conditional_edges(
    "execute_task",
    next_step
)

workflow.add_edge("generate_response", END)

# Compile the graph
autonomous_agent = workflow.compile()

# Run the autonomous agent
async def run_autonomous_agent():
    result = await autonomous_agent.ainvoke({
        "messages": [
            HumanMessage(content="Research the latest advancements in renewable energy.")
        ],
        "memory": {},
        "task_queue": [],
        "task_results": {}
    })

    print("Final Response:", result["messages"][-1].content)

import asyncio
asyncio.run(run_autonomous_agent())`,
          description: 'Building an autonomous agent with task planning, execution, and memory'
        }]
      }
    ]
  }
};
