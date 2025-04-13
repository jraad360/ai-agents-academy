import { Module } from '../../../types';

export const langgraphFundamentalsModule: Module = {
  id: 'langgraph-fundamentals',
  title: 'LangGraph Fundamentals',
  description: 'Understanding the core concepts of LangGraph for building stateful agent workflows',
  duration: '2 hours',
  level: 'intermediate',
  content: {
    sections: [
      {
        title: 'Graph-Based Workflows',
        content: `LangGraph is a framework for building stateful, multi-step AI applications using a graph-based approach. It provides a structured way to orchestrate complex agent workflows with explicit state management.

Key concepts of graph-based workflows:

• Core Components
  - Nodes: Functions that perform specific tasks (reasoning, tool use, etc.)
  - Edges: Define the flow between nodes
  - State: Shared data structure representing the current application state
  - Channels: Named components of the state that can be updated independently

• Benefits
  - Explicit control flow and state management
  - Support for complex, non-linear workflows
  - Ability to implement cycles and conditional branching
  - Improved debugging and observability

• Use Cases
  - Multi-step reasoning processes
  - Conversational agents with memory
  - Multi-agent systems with specialized roles
  - Human-in-the-loop workflows

LangGraph is built on the concept of message passing, where nodes communicate by sending messages along edges. This approach, inspired by Google's Pregel system, allows for complex, stateful workflows that can evolve over time.

Learn more about LangGraph's architecture in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/low_level/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";

// Define the state interface
interface GraphState {
  input: string;
  intermediate_result?: string;
  output?: string;
}

// Initialize the language model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0
});

// Create a simple graph
const workflow = new StateGraph<GraphState>({
  channels: {
    input: {
      value: (x: string) => x,
      default: () => ""
    },
    intermediate_result: {
      value: (x: string) => x,
      default: () => undefined
    },
    output: {
      value: (x: string) => x,
      default: () => undefined
    }
  }
});

// Define node functions
async function processInput(state: GraphState) {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that summarizes text."],
    ["human", "Summarize the following text in one sentence: {input}"]
  ]);

  const chain = RunnableSequence.from([
    prompt,
    model,
    new StringOutputParser()
  ]);

  const result = await chain.invoke({ input: state.input });

  return { intermediate_result: result };
}

async function enhanceOutput(state: GraphState) {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that enhances text."],
    ["human", "Add more detail to the following summary: {intermediate_result}"]
  ]);

  const chain = RunnableSequence.from([
    prompt,
    model,
    new StringOutputParser()
  ]);

  const result = await chain.invoke({
    intermediate_result: state.intermediate_result
  });

  return { output: result };
}

// Add nodes to the graph
workflow.addNode("process_input", processInput);
workflow.addNode("enhance_output", enhanceOutput);

// Define the edges
workflow.setEntryPoint("process_input");
workflow.addEdge("process_input", "enhance_output");
workflow.addEdge("enhance_output", END);

// Compile the graph
const app = workflow.compile();

// Run the graph
async function runGraph() {
  const result = await app.invoke({
    input: "LangGraph is a framework for building stateful applications with LLMs. It provides a flexible way to compose chains and agents into graphs with memory and persistence."
  });

  console.log("Final output:", result.output);
}

runGraph();`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from typing import TypedDict, Optional

# Define the state type
class GraphState(TypedDict):
    input: str
    intermediate_result: Optional[str]
    output: Optional[str]

# Initialize the language model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Create a simple graph
workflow = StateGraph(GraphState)

# Define node functions
async def process_input(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes text."),
        ("human", "Summarize the following text in one sentence: {input}")
    ])

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    result = await chain.ainvoke({"input": state["input"]})

    return {"intermediate_result": result}

async def enhance_output(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that enhances text."),
        ("human", "Add more detail to the following summary: {intermediate_result}")
    ])

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    result = await chain.ainvoke({
        "intermediate_result": state["intermediate_result"]
    })

    return {"output": result}

# Add nodes to the graph
workflow.add_node("process_input", process_input)
workflow.add_node("enhance_output", enhance_output)

# Define the edges
workflow.set_entry_point("process_input")
workflow.add_edge("process_input", "enhance_output")
workflow.add_edge("enhance_output", END)

# Compile the graph
app = workflow.compile()

# Run the graph
async def run_graph():
    result = await app.ainvoke({
        "input": "LangGraph is a framework for building stateful applications with LLMs. It provides a flexible way to compose chains and agents into graphs with memory and persistence."
    })

    print("Final output:", result["output"])

import asyncio
asyncio.run(run_graph())`,
          description: 'Creating a simple LangGraph workflow with multiple processing steps'
        }]
      },
      {
        title: 'State Management',
        content: `State management is a core concept in LangGraph that allows agents to maintain context and share information across different steps of a workflow. Proper state management is essential for building complex, stateful applications.

Key concepts of state management:

• State Schema
  - Defines the structure of the shared state
  - Can use TypedDict (TypeScript/Python) or Pydantic models (Python)
  - Specifies the types and default values for state channels
  - Provides type safety and validation

• Channels
  - Named components of the state that can be updated independently
  - Each channel has its own reducer function for applying updates
  - Allows for fine-grained control over how state is updated
  - Enables parallel updates to different parts of the state

• Reducers
  - Functions that determine how updates are applied to channels
  - Default reducer simply overwrites the channel value
  - Custom reducers can implement more complex logic (append, merge, etc.)
  - Essential for handling lists, maps, and other complex data structures

• State Updates
  - Nodes return partial state updates (only the channels they modify)
  - Updates are applied to the current state using reducers
  - The updated state is passed to the next node(s) in the workflow
  - Enables incremental state evolution throughout the graph execution

Learn more about state management in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/low_level/#state).`,
        codeExamples: [{
          typescript: `import { StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

// Define a state interface with different types of channels
interface AgentState {
  // Simple string channel (default reducer: overwrite)
  input: string;

  // Array channel with custom reducer (append)
  messages: Array<HumanMessage | AIMessage>;

  // Optional result channel
  result?: string;

  // Object channel with nested data
  metadata: {
    user_id: string;
    session_id: string;
    timestamps: {
      start: number;
      last_update?: number;
    };
  };
}

// Custom reducer for the messages channel
function addMessages(currentMessages: Array<HumanMessage | AIMessage>, newMessages: Array<HumanMessage | AIMessage>) {
  return [...currentMessages, ...newMessages];
}

// Create a graph with the defined state
const workflow = new StateGraph<AgentState>({
  channels: {
    input: {
      value: (x: string) => x,
      default: () => ""
    },
    messages: {
      value: (x: Array<HumanMessage | AIMessage>) => x,
      default: () => [],
      reducer: addMessages  // Custom reducer for appending messages
    },
    result: {
      value: (x: string) => x,
      default: () => undefined
    },
    metadata: {
      value: (x: any) => x,
      default: () => ({
        user_id: "",
        session_id: "",
        timestamps: {
          start: Date.now()
        }
      })
    }
  }
});

// Initialize the language model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0
});

// Define a node that updates multiple state channels
async function processInput(state: AgentState) {
  // Create a prompt
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant."]
  ]);

  // Add the user message to the messages channel
  const userMessage = new HumanMessage(state.input);

  // Generate a response
  const response = await model.invoke([...state.messages, userMessage]);

  // Update multiple channels in the state
  return {
    // Append new messages to the messages channel
    messages: [userMessage, response],

    // Set the result channel
    result: response.content as string,

    // Update nested fields in the metadata channel
    metadata: {
      ...state.metadata,
      timestamps: {
        ...state.metadata.timestamps,
        last_update: Date.now()
      }
    }
  };
}

// Add the node to the graph
workflow.addNode("process_input", processInput);

// Set up the rest of the graph...
// workflow.setEntryPoint("process_input");
// workflow.addEdge("process_input", ...);

// Compile the graph
const app = workflow.compile();

// Example of invoking the graph with initial state
async function runGraph() {
  const result = await app.invoke({
    input: "What is LangGraph?",
    messages: [],
    metadata: {
      user_id: "user-123",
      session_id: "session-456",
      timestamps: {
        start: Date.now()
      }
    }
  });

  console.log("Final state:", result);
}`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
import time
from operator import add  # Used as a reducer for appending lists

# Define a state type with different types of channels
class Timestamps(TypedDict):
    start: float
    last_update: Optional[float]

class Metadata(TypedDict):
    user_id: str
    session_id: str
    timestamps: Timestamps

class AgentState(TypedDict):
    # Simple string channel (default reducer: overwrite)
    input: str

    # Array channel with custom reducer (append)
    messages: Annotated[List[HumanMessage | AIMessage], add]

    # Optional result channel
    result: Optional[str]

    # Object channel with nested data
    metadata: Metadata

# Create a graph with the defined state
workflow = StateGraph(AgentState)

# Initialize the language model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Define a node that updates multiple state channels
async def process_input(state: AgentState) -> Dict[str, Any]:
    # Create a prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant.")
    ])

    # Add the user message to the messages channel
    user_message = HumanMessage(content=state["input"])

    # Generate a response
    response = await model.ainvoke(state["messages"] + [user_message])

    # Update multiple channels in the state
    return {
        # Append new messages to the messages channel
        "messages": [user_message, response],

        # Set the result channel
        "result": response.content,

        # Update nested fields in the metadata channel
        "metadata": {
            **state["metadata"],
            "timestamps": {
                **state["metadata"]["timestamps"],
                "last_update": time.time()
            }
        }
    }

# Add the node to the graph
workflow.add_node("process_input", process_input)

# Set up the rest of the graph...
# workflow.set_entry_point("process_input")
# workflow.add_edge("process_input", ...)

# Compile the graph
app = workflow.compile()

# Example of invoking the graph with initial state
async def run_graph():
    result = await app.ainvoke({
        "input": "What is LangGraph?",
        "messages": [],
        "metadata": {
            "user_id": "user-123",
            "session_id": "session-456",
            "timestamps": {
                "start": time.time()
            }
        }
    })

    print("Final state:", result)`,
          description: 'Managing complex state with different channel types and custom reducers'
        }]
      },
      {
        title: 'Streaming and Events',
        content: `LangGraph provides robust support for streaming, allowing you to receive incremental updates as your agent workflow progresses. This is particularly valuable for long-running processes and real-time applications.

Key concepts of streaming and events:

• Streaming Modes
  - State Streaming: Receive updates to the graph state after each node execution
  - Token Streaming: Stream individual tokens from LLM responses
  - Combined Streaming: Get both state updates and token streams

• Benefits
  - Improved user experience with real-time feedback
  - Visibility into intermediate steps and reasoning
  - Ability to display progress for long-running tasks
  - Early error detection and debugging

• Implementation Approaches
  - Using the stream() method instead of invoke()
  - Handling stream events with callbacks or async iterators
  - Configuring which state channels to stream
  - Managing stream lifecycle with proper error handling

• Advanced Features
  - Streaming from nested components and subgraphs
  - Filtering and transforming stream events
  - Handling backpressure in high-throughput scenarios
  - Integrating with WebSockets and SSE for web applications

Streaming is particularly important for maintaining user engagement during complex, multi-step agent workflows that might otherwise appear unresponsive.

Learn more about streaming in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/streaming/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";

// Define the state interface
interface GraphState {
  input: string;
  intermediate_result?: string;
  output?: string;
}

// Initialize the language model with streaming enabled
const model = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0,
  streaming: true
});

// Create a graph
const workflow = new StateGraph<GraphState>({
  channels: {
    input: {
      value: (x: string) => x,
      default: () => ""
    },
    intermediate_result: {
      value: (x: string) => x,
      default: () => undefined
    },
    output: {
      value: (x: string) => x,
      default: () => undefined
    }
  }
});

// Define node functions
async function generateContent(state: GraphState) {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that generates detailed content."],
    ["human", "Write a detailed explanation about: {input}"]
  ]);

  const chain = RunnableSequence.from([
    prompt,
    model,
    new StringOutputParser()
  ]);

  const result = await chain.invoke({ input: state.input });

  return { intermediate_result: result };
}

async function summarizeContent(state: GraphState) {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that summarizes content."],
    ["human", "Summarize the following content in a few sentences: {intermediate_result}"]
  ]);

  const chain = RunnableSequence.from([
    prompt,
    model,
    new StringOutputParser()
  ]);

  const result = await chain.invoke({
    intermediate_result: state.intermediate_result
  });

  return { output: result };
}

// Add nodes to the graph
workflow.addNode("generate_content", generateContent);
workflow.addNode("summarize_content", summarizeContent);

// Define the edges
workflow.setEntryPoint("generate_content");
workflow.addEdge("generate_content", "summarize_content");
workflow.addEdge("summarize_content", END);

// Compile the graph
const app = workflow.compile();

// Stream the graph execution
async function streamGraph() {
  const stream = await app.stream({
    input: "The history and evolution of artificial intelligence"
  });

  // Process the stream
  for await (const chunk of stream) {
    if (chunk.kind === "state") {
      // State update
      console.log("State update:", chunk.state);
    } else if (chunk.kind === "llm") {
      // Token from LLM
      process.stdout.write(chunk.data);
    }
  }
}

streamGraph();`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from typing import TypedDict, Optional, AsyncIterator, Dict, Any, Union
import sys

# Define the state type
class GraphState(TypedDict):
    input: str
    intermediate_result: Optional[str]
    output: Optional[str]

# Initialize the language model with streaming enabled
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True
)

# Create a graph
workflow = StateGraph(GraphState)

# Define node functions
async def generate_content(state: GraphState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates detailed content."),
        ("human", "Write a detailed explanation about: {input}")
    ])

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    result = await chain.ainvoke({"input": state["input"]})

    return {"intermediate_result": result}

async def summarize_content(state: GraphState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes content."),
        ("human", "Summarize the following content in a few sentences: {intermediate_result}")
    ])

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    result = await chain.ainvoke({
        "intermediate_result": state["intermediate_result"]
    })

    return {"output": result}

# Add nodes to the graph
workflow.add_node("generate_content", generate_content)
workflow.add_node("summarize_content", summarize_content)

# Define the edges
workflow.set_entry_point("generate_content")
workflow.add_edge("generate_content", "summarize_content")
workflow.add_edge("summarize_content", END)

# Compile the graph
app = workflow.compile()

# Stream the graph execution
async def stream_graph():
    stream = await app.astream({
        "input": "The history and evolution of artificial intelligence"
    })

    # Process the stream
    async for chunk in stream:
        if chunk.get("kind") == "state":
            # State update
            print(f"State update: {chunk['state']}")
        elif chunk.get("kind") == "llm":
            # Token from LLM
            sys.stdout.write(chunk["data"])
            sys.stdout.flush()

import asyncio
asyncio.run(stream_graph())`,
          description: 'Implementing streaming in a LangGraph workflow to receive real-time updates'
        }]
      },
      {
        title: 'Human-in-the-Loop',
        content: `Human-in-the-loop (HITL) workflows allow agents to pause execution and request human input or approval before continuing. This capability is essential for applications where human oversight, validation, or collaboration is required.

Key concepts of human-in-the-loop:

• Interruption Mechanisms
  - Using the interrupt() function to pause graph execution
  - Specifying what information to send to the human
  - Waiting for human input before resuming
  - Handling timeouts and cancellations

• Use Cases
  - Approval workflows for critical decisions
  - Human validation of generated content
  - Collaborative problem-solving
  - Handling edge cases and exceptions

• Implementation Approaches
  - Explicit interruption nodes in the graph
  - Conditional interruption based on confidence scores
  - Providing context and options to the human
  - Resuming execution with human input

• Integration Patterns
  - Web interfaces with WebSockets or SSE
  - Chat applications with turn-taking
  - Email or notification-based workflows
  - Mobile app integrations

Human-in-the-loop capabilities are particularly valuable for high-stakes applications where errors could be costly, or in domains where human expertise and judgment remain essential.

Learn more about human-in-the-loop workflows in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { interrupt } from "@langchain/langgraph/types";
import { Command } from "@langchain/langgraph/types";

// Define the state interface
interface GraphState {
  input: string;
  draft?: string;
  feedback?: string;
  final_output?: string;
}

// Initialize the language model
const model = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

// Create a graph
const workflow = new StateGraph<GraphState>({
  channels: {
    input: {
      value: (x: string) => x,
      default: () => ""
    },
    draft: {
      value: (x: string) => x,
      default: () => undefined
    },
    feedback: {
      value: (x: string) => x,
      default: () => undefined
    },
    final_output: {
      value: (x: string) => x,
      default: () => undefined
    }
  }
});

// Define node functions
async function generateDraft(state: GraphState) {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that drafts content."],
    ["human", "Create a draft for: {input}"]
  ]);

  const chain = RunnableSequence.from([
    prompt,
    model,
    new StringOutputParser()
  ]);

  const result = await chain.invoke({ input: state.input });

  return { draft: result };
}

async function humanReview(state: GraphState) {
  // Interrupt the graph and wait for human feedback
  const feedback = interrupt({
    // Information to send to the human
    draft: state.draft,
    message: "Please review this draft and provide feedback.",
    // Optional: Provide suggested actions
    actions: ["approve", "revise", "reject"]
  });

  // This code will only execute after the human provides input
  return { feedback };
}

async function finalizeDraft(state: GraphState) {
  // Check if the human approved the draft
  if (state.feedback === "approve") {
    return { final_output: state.draft };
  }

  // If the human requested revisions, update the draft
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that revises content based on feedback."],
    ["human", "Original draft: {draft}\n\nFeedback: {feedback}\n\nPlease revise the draft based on this feedback."]
  ]);

  const chain = RunnableSequence.from([
    prompt,
    model,
    new StringOutputParser()
  ]);

  const result = await chain.invoke({
    draft: state.draft,
    feedback: state.feedback
  });

  return { final_output: result };
}

// Add nodes to the graph
workflow.addNode("generate_draft", generateDraft);
workflow.addNode("human_review", humanReview);
workflow.addNode("finalize_draft", finalizeDraft);

// Define the edges
workflow.setEntryPoint("generate_draft");
workflow.addEdge("generate_draft", "human_review");
workflow.addEdge("human_review", "finalize_draft");
workflow.addEdge("finalize_draft", END);

// Compile the graph
const app = workflow.compile();

// Example of running the graph with human-in-the-loop
async function runGraphWithHuman() {
  // Start the graph execution
  let result = await app.invoke({
    input: "Write a short blog post about AI safety"
  });

  // Check if the graph is waiting for human input
  if (result.feedback === undefined) {
    console.log("Draft generated:", result.draft);
    console.log("Waiting for human feedback...");

    // In a real application, you would get input from a user interface
    const humanFeedback = "Please make it more accessible to non-technical readers.";

    // Resume the graph with human input
    result = await app.invoke({
      ...result,
      feedback: humanFeedback
    });
  }

  console.log("Final output:", result.final_output);
}

runGraphWithHuman();`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from typing import TypedDict, Optional, Dict, Any, List
from langgraph.types import interrupt, Command

# Define the state type
class GraphState(TypedDict):
    input: str
    draft: Optional[str]
    feedback: Optional[str]
    final_output: Optional[str]

# Initialize the language model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Create a graph
workflow = StateGraph(GraphState)

# Define node functions
async def generate_draft(state: GraphState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that drafts content."),
        ("human", "Create a draft for: {input}")
    ])

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    result = await chain.ainvoke({"input": state["input"]})

    return {"draft": result}

def human_review(state: GraphState) -> Dict[str, Any]:
    # Interrupt the graph and wait for human feedback
    feedback = interrupt({
        # Information to send to the human
        "draft": state["draft"],
        "message": "Please review this draft and provide feedback.",
        # Optional: Provide suggested actions
        "actions": ["approve", "revise", "reject"]
    })

    # This code will only execute after the human provides input
    return {"feedback": feedback}

async def finalize_draft(state: GraphState) -> Dict[str, Any]:
    # Check if the human approved the draft
    if state["feedback"] == "approve":
        return {"final_output": state["draft"]}

    # If the human requested revisions, update the draft
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that revises content based on feedback."),
        ("human", "Original draft: {draft}\n\nFeedback: {feedback}\n\nPlease revise the draft based on this feedback.")
    ])

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    result = await chain.ainvoke({
        "draft": state["draft"],
        "feedback": state["feedback"]
    })

    return {"final_output": result}

# Add nodes to the graph
workflow.add_node("generate_draft", generate_draft)
workflow.add_node("human_review", human_review)
workflow.add_node("finalize_draft", finalize_draft)

# Define the edges
workflow.set_entry_point("generate_draft")
workflow.add_edge("generate_draft", "human_review")
workflow.add_edge("human_review", "finalize_draft")
workflow.add_edge("finalize_draft", END)

# Compile the graph
app = workflow.compile()

# Example of running the graph with human-in-the-loop
async def run_graph_with_human():
    # Start the graph execution
    result = await app.ainvoke({
        "input": "Write a short blog post about AI safety"
    })

    # Check if the graph is waiting for human input
    if result.get("feedback") is None:
        print(f"Draft generated: {result['draft']}")
        print("Waiting for human feedback...")

        # In a real application, you would get input from a user interface
        human_feedback = "Please make it more accessible to non-technical readers."

        # Resume the graph with human input
        result = await app.ainvoke({
            **result,
            "feedback": human_feedback
        })

    print(f"Final output: {result['final_output']}")

import asyncio
asyncio.run(run_graph_with_human())`,
          description: 'Implementing a human-in-the-loop workflow with review and approval steps'
        }]
      }
    ]
  }
};
