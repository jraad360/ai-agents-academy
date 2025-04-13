import { Module } from '../../../types';

export const multiAgentSystemsModule: Module = {
  id: 'multi-agent-systems',
  title: 'Multi-Agent Systems',
  description: 'Building collaborative AI systems with multiple specialized agents',
  duration: '2 hours',
  level: 'advanced',
  content: {
    sections: [
      {
        title: 'Multi-Agent Architecture',
        content: `Multi-agent systems consist of multiple AI agents working together to solve complex problems. These systems leverage the strengths of specialized agents to achieve goals that would be difficult for a single agent to accomplish.

Key concepts of multi-agent architecture:

• Core Components
  - Agent Roles: Specialized functions for each agent (researcher, coder, critic, etc.)
  - Communication Protocols: How agents exchange information
  - Coordination Mechanisms: How agent activities are orchestrated
  - Shared State: Information accessible to all agents

• Architectural Patterns
  - Supervisor Pattern: Central agent delegates tasks and coordinates others
  - Peer-to-Peer: Agents communicate directly without central coordination
  - Assembly Line: Sequential processing with specialized agents
  - Team of Experts: Agents with complementary skills collaborate on tasks

• Implementation Considerations
  - Agent Specialization: Tailoring prompts and tools for specific roles
  - State Management: Sharing context between agents
  - Error Handling: Recovering from agent failures
  - Termination Conditions: Determining when the task is complete

Multi-agent systems are particularly valuable for complex tasks requiring diverse skills, such as research, coding, content creation, and decision-making.

Learn more about multi-agent architectures in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/multi_agent/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage } from "@langchain/core/messages";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { tool } from "@langchain/core/tools";

// Define the state interface
interface TeamState {
  messages: any[];
  next?: string;
}

// Initialize the language model
const llm = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

// Create tools for the agents
const searchTool = new TavilySearchResults({
  maxResults: 3
});

// Create a coding tool
const codingTool = tool(
  async ({ code }) => {
    // In a real implementation, this would execute code safely
    console.log("Executing code:", code);
    return "Code execution result would appear here";
  },
  {
    name: "execute_code",
    description: "Execute code and return the result",
    schema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "The code to execute"
        }
      },
      required: ["code"]
    }
  }
);

// Define the supervisor agent
const supervisorPrompt = ChatPromptTemplate.fromMessages([
  ["system", \`You are a supervisor tasked with managing a conversation between the
following workers: researcher, coder. Given the following user request,
respond with the worker to act next. Each worker will perform a
task and respond with their results and status. When finished,
respond with FINISH.\`],
  ["human", "{input}"],
  ["human", "Current conversation: {messages}"]
]);

// Create the supervisor node
async function supervisorNode(state: TeamState) {
  const response = await llm.invoke(
    await supervisorPrompt.formatMessages({
      input: state.messages[0].content,
      messages: state.messages.slice(1).map(m => (m.name || 'user') + ': ' + m.content).join("\\n")
    })
  );

  // Parse the response to determine the next agent
  const content = response.content as string;
  let next = "";

  if (content.includes("FINISH")) {
    next = END;
  } else if (content.toLowerCase().includes("researcher")) {
    next = "researcher";
  } else if (content.toLowerCase().includes("coder")) {
    next = "coder";
  } else {
    next = END;
  }

  return { ...state, next };
}

// Define the researcher agent
const researcherPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a research agent that finds information on topics. Use the search tool to find relevant information."],
  ["human", "{input}"]
]);

// Create the researcher node
async function researcherNode(state: TeamState) {
  const researchChain = researcherPrompt
    .pipe(llm.bind({ tools: [searchTool] }))
    .pipe(new StringOutputParser());

  const result = await researchChain.invoke({
    input: state.messages[0].content
  });

  return {
    messages: [
      ...state.messages,
      new HumanMessage({
        content: result,
        name: "researcher"
      })
    ],
    next: "supervisor"
  };
}

// Define the coder agent
const coderPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a coding agent that writes and executes code to solve problems."],
  ["human", "{input}"]
]);

// Create the coder node
async function coderNode(state: TeamState) {
  const codingChain = coderPrompt
    .pipe(llm.bind({ tools: [codingTool] }))
    .pipe(new StringOutputParser());

  const result = await codingChain.invoke({
    input: state.messages[0].content
  });

  return {
    messages: [
      ...state.messages,
      new HumanMessage({
        content: result,
        name: "coder"
      })
    ],
    next: "supervisor"
  };
}

// Create the graph
const workflow = new StateGraph<TeamState>({
  channels: {
    messages: {
      value: (x: any[]) => x,
      default: () => []
    },
    next: {
      value: (x: string) => x,
      default: () => undefined
    }
  }
});

// Add nodes to the graph
workflow.addNode("supervisor", supervisorNode);
workflow.addNode("researcher", researcherNode);
workflow.addNode("coder", coderNode);

// Define the edges
workflow.setEntryPoint("supervisor");
workflow.addConditionalEdges(
  "supervisor",
  (state) => state.next
);
workflow.addEdge("researcher", "supervisor");
workflow.addEdge("coder", "supervisor");

// Compile the graph
const multiAgentSystem = workflow.compile();

// Example usage
async function runMultiAgentSystem() {
  const result = await multiAgentSystem.invoke({
    messages: [
      new HumanMessage("Research the latest advancements in quantum computing and write a Python function to simulate a simple quantum gate.")
    ]
  });

  console.log("Final conversation:");
  result.messages.forEach(message => {
    console.log((message.name || 'user') + ': ' + message.content);
  });
}`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import TypedDict, List, Dict, Any, Optional

# Define the state type
class TeamState(TypedDict):
    messages: List[Any]
    next: Optional[str]

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Create tools for the agents
search_tool = TavilySearchResults(
    max_results=3
)

# Create a coding tool
@tool
def execute_code(code: str) -> str:
    """Execute code and return the result."""
    # In a real implementation, this would execute code safely
    print(f"Executing code: {code}")
    return "Code execution result would appear here"

# Define the supervisor agent
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a supervisor tasked with managing a conversation between the
following workers: researcher, coder. Given the following user request,
respond with the worker to act next. Each worker will perform a
task and respond with their results and status. When finished,
respond with FINISH."""),
    ("human", "{input}"),
    ("human", "Current conversation: {messages}")
])

# Create the supervisor node
async def supervisor_node(state: TeamState) -> TeamState:
    response = await llm.ainvoke(
        supervisor_prompt.format_messages(
            input=state["messages"][0].content,
            messages="\n".join([f"{m.name or 'user'}: {m.content}" for m in state["messages"][1:]])
        )
    )

    # Parse the response to determine the next agent
    content = response.content
    next_agent = None

    if "FINISH" in content:
        next_agent = END
    elif "researcher" in content.lower():
        next_agent = "researcher"
    elif "coder" in content.lower():
        next_agent = "coder"
    else:
        next_agent = END

    return {**state, "next": next_agent}

# Define the researcher agent
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research agent that finds information on topics. Use the search tool to find relevant information."),
    ("human", "{input}")
])

# Create the researcher node
async def researcher_node(state: TeamState) -> TeamState:
    research_chain = (
        researcher_prompt
        | llm.bind_tools([search_tool])
        | StrOutputParser()
    )

    result = await research_chain.ainvoke({
        "input": state["messages"][0].content
    })

    return {
        "messages": [
            *state["messages"],
            HumanMessage(content=result, name="researcher")
        ],
        "next": "supervisor"
    }

# Define the coder agent
coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a coding agent that writes and executes code to solve problems."),
    ("human", "{input}")
])

# Create the coder node
async def coder_node(state: TeamState) -> TeamState:
    coding_chain = (
        coder_prompt
        | llm.bind_tools([execute_code])
        | StrOutputParser()
    )

    result = await coding_chain.ainvoke({
        "input": state["messages"][0].content
    })

    return {
        "messages": [
            *state["messages"],
            HumanMessage(content=result, name="coder")
        ],
        "next": "supervisor"
    }

# Create the graph
workflow = StateGraph(TeamState)

# Add nodes to the graph
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)

# Define the edges
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"]
)
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("coder", "supervisor")

# Compile the graph
multi_agent_system = workflow.compile()

# Example usage
async def run_multi_agent_system():
    result = await multi_agent_system.ainvoke({
        "messages": [
            HumanMessage(content="Research the latest advancements in quantum computing and write a Python function to simulate a simple quantum gate.")
        ]
    })

    print("Final conversation:")
    for message in result["messages"]:
        print(f"{message.name or 'user'}: {message.content}")

import asyncio
asyncio.run(run_multi_agent_system())`,
          description: 'Building a multi-agent system with supervisor, researcher, and coder agents'
        }]
      },
      {
        title: 'Collaboration Patterns',
        content: `Multi-agent systems can employ various collaboration patterns to effectively coordinate agent activities. These patterns determine how agents interact, share information, and make decisions collectively.

Key collaboration patterns:

• Supervisor Pattern
  - Central agent coordinates and delegates tasks to specialized agents
  - Supervisor evaluates agent outputs and makes routing decisions
  - Provides clear control flow and responsibility hierarchy
  - Well-suited for complex workflows with distinct specialist roles

• Peer-to-Peer Pattern
  - Agents communicate directly without central coordination
  - Each agent can initiate interactions with other agents
  - More flexible and resilient to individual agent failures
  - Requires sophisticated communication protocols

• Assembly Line Pattern
  - Sequential processing with specialized agents
  - Each agent performs a specific transformation on the input
  - Output of one agent becomes input to the next
  - Efficient for well-defined, linear workflows

• Team of Experts Pattern
  - Multiple specialized agents collaborate on a shared task
  - Agents contribute based on their expertise
  - May include voting or consensus mechanisms
  - Effective for complex problems requiring diverse perspectives

Choosing the right collaboration pattern depends on the task complexity, the need for oversight, and the desired balance between efficiency and flexibility.

Learn more about collaboration patterns in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/multi_agent/).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { RunnableSequence } from "@langchain/core/runnables";

// Define the state interface for a peer-to-peer multi-agent system
interface PeerState {
  messages: any[];
  current_agent?: string;
  task_complete: boolean;
}

// Initialize the language model
const llm = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

// Define agent roles and their system prompts
const agentRoles = {
  planner: "You are a planning agent that breaks down complex tasks into steps. You create clear, actionable plans.",
  researcher: "You are a research agent that finds and summarizes information. You provide comprehensive and accurate information.",
  writer: "You are a writing agent that creates polished content. You write clear, engaging, and well-structured text.",
  critic: "You are a critical thinking agent that evaluates content. You identify logical flaws, missing information, and areas for improvement."
};

// Create a function to determine if a task is complete
function isTaskComplete(state: PeerState) {
  // In a real implementation, this would have more sophisticated logic
  // For example, checking if all required steps are completed
  return state.task_complete;
}

// Create a function to determine the next agent
function determineNextAgent(state: PeerState) {
  const lastMessage = state.messages[state.messages.length - 1];
  const content = lastMessage.content.toLowerCase();

  // Simple routing logic based on message content
  if (content.includes("need more information") || content.includes("research")) {
    return "researcher";
  } else if (content.includes("write") || content.includes("draft")) {
    return "writer";
  } else if (content.includes("review") || content.includes("evaluate")) {
    return "critic";
  } else if (content.includes("plan") || content.includes("steps")) {
    return "planner";
  } else if (content.includes("complete") || content.includes("finished")) {
    return "end";
  }

  // Default to planner if no clear direction
  return "planner";
}

// Create agent nodes
function createAgentNode(role: string) {
  return async (state: PeerState) => {
    // Create a prompt for this agent
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", agentRoles[role]],
      ["human", "Task: {task}"],
      ["human", "Conversation history: {history}"],
      ["human", "Based on the above, perform your role as the {role} agent. If you believe the task is complete, say 'TASK COMPLETE'."]
    ]);

    // Create a chain for this agent
    const chain = RunnableSequence.from([
      prompt,
      llm,
      new StringOutputParser()
    ]);

    // Get the task from the first message
    const task = state.messages[0].content;

    // Format the conversation history
    const history = state.messages.slice(1).map(m =>
      (m.name || 'user') + ': ' + m.content
    ).join("\n");

    // Invoke the chain
    const result = await chain.invoke({
      task,
      history,
      role
    });

    // Check if the task is complete
    const taskComplete = result.includes("TASK COMPLETE");

    // Return the updated state
    return {
      messages: [
        ...state.messages,
        new AIMessage({
          content: result,
          name: role
        })
      ],
      task_complete: taskComplete
    };
  };
}

// Create the graph
const workflow = new StateGraph<PeerState>({
  channels: {
    messages: {
      value: (x: any[]) => x,
      default: () => []
    },
    current_agent: {
      value: (x: string) => x,
      default: () => undefined
    },
    task_complete: {
      value: (x: boolean) => x,
      default: () => false
    }
  }
});

// Add nodes to the graph
workflow.addNode("planner", createAgentNode("planner"));
workflow.addNode("researcher", createAgentNode("researcher"));
workflow.addNode("writer", createAgentNode("writer"));
workflow.addNode("critic", createAgentNode("critic"));

// Define the router node
async function routerNode(state: PeerState) {
  if (state.task_complete) {
    return { ...state, current_agent: "end" };
  }

  const nextAgent = determineNextAgent(state);
  return { ...state, current_agent: nextAgent };
}

workflow.addNode("router", routerNode);

// Define the edges
workflow.setEntryPoint("router");

// Add conditional edges from the router
workflow.addConditionalEdges(
  "router",
  (state) => state.current_agent
);

// Add edges from each agent back to the router
workflow.addEdge("planner", "router");
workflow.addEdge("researcher", "router");
workflow.addEdge("writer", "router");
workflow.addEdge("critic", "router");

// Compile the graph
const peerAgentSystem = workflow.compile();

// Example usage
async function runPeerAgentSystem() {
  const result = await peerAgentSystem.invoke({
    messages: [
      new HumanMessage("Create a comprehensive blog post about the impact of artificial intelligence on healthcare.")
    ],
    task_complete: false
  });

  console.log("Final conversation:");
  result.messages.forEach(message => {
    console.log((message.name || 'user') + ': ' + message.content);
  });
}

runPeerAgentSystem();`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableSequence
from typing import TypedDict, List, Dict, Any, Optional

# Define the state type for a peer-to-peer multi-agent system
class PeerState(TypedDict):
    messages: List[Any]
    current_agent: Optional[str]
    task_complete: bool

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Define agent roles and their system prompts
agent_roles = {
    "planner": "You are a planning agent that breaks down complex tasks into steps. You create clear, actionable plans.",
    "researcher": "You are a research agent that finds and summarizes information. You provide comprehensive and accurate information.",
    "writer": "You are a writing agent that creates polished content. You write clear, engaging, and well-structured text.",
    "critic": "You are a critical thinking agent that evaluates content. You identify logical flaws, missing information, and areas for improvement."
}

# Create a function to determine if a task is complete
def is_task_complete(state: PeerState) -> bool:
    # In a real implementation, this would have more sophisticated logic
    # For example, checking if all required steps are completed
    return state["task_complete"]

# Create a function to determine the next agent
def determine_next_agent(state: PeerState) -> str:
    last_message = state["messages"][-1]
    content = last_message.content.lower()

    # Simple routing logic based on message content
    if "need more information" in content or "research" in content:
        return "researcher"
    elif "write" in content or "draft" in content:
        return "writer"
    elif "review" in content or "evaluate" in content:
        return "critic"
    elif "plan" in content or "steps" in content:
        return "planner"
    elif "complete" in content or "finished" in content:
        return "end"

    # Default to planner if no clear direction
    return "planner"

# Create agent nodes
def create_agent_node(role: str):
    async def agent_node(state: PeerState) -> PeerState:
        # Create a prompt for this agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_roles[role]),
            ("human", "Task: {task}"),
            ("human", "Conversation history: {history}"),
            ("human", "Based on the above, perform your role as the {role} agent. If you believe the task is complete, say 'TASK COMPLETE'.")
        ])

        # Create a chain for this agent
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        # Get the task from the first message
        task = state["messages"][0].content

        # Format the conversation history
        history = "\n".join([
            (m.name or 'user') + ': ' + m.content
            for m in state["messages"][1:]
        ])

        # Invoke the chain
        result = await chain.ainvoke({
            "task": task,
            "history": history,
            "role": role
        })

        # Check if the task is complete
        task_complete = "TASK COMPLETE" in result

        # Return the updated state
        return {
            "messages": [
                *state["messages"],
                AIMessage(content=result, name=role)
            ],
            "task_complete": task_complete
        }

    return agent_node

# Create the graph
workflow = StateGraph(PeerState)

# Add nodes to the graph
workflow.add_node("planner", create_agent_node("planner"))
workflow.add_node("researcher", create_agent_node("researcher"))
workflow.add_node("writer", create_agent_node("writer"))
workflow.add_node("critic", create_agent_node("critic"))

# Define the router node
async def router_node(state: PeerState) -> PeerState:
    if state["task_complete"]:
        return {**state, "current_agent": "end"}

    next_agent = determine_next_agent(state)
    return {**state, "current_agent": next_agent}

workflow.add_node("router", router_node)

# Define the edges
workflow.set_entry_point("router")

# Add conditional edges from the router
workflow.add_conditional_edges(
    "router",
    lambda state: state["current_agent"]
)

# Add edges from each agent back to the router
workflow.add_edge("planner", "router")
workflow.add_edge("researcher", "router")
workflow.add_edge("writer", "router")
workflow.add_edge("critic", "router")

# Compile the graph
peer_agent_system = workflow.compile()

# Example usage
async def run_peer_agent_system():
    result = await peer_agent_system.ainvoke({
        "messages": [
            HumanMessage(content="Create a comprehensive blog post about the impact of artificial intelligence on healthcare.")
        ],
        "task_complete": False
    })

    print("Final conversation:")
    for message in result["messages"]:
        print((message.name or 'user') + ': ' + message.content)

import asyncio
asyncio.run(run_peer_agent_system())`,
          description: 'Implementing a peer-to-peer multi-agent system with dynamic routing between specialized agents'
        }]
      },
      {
        title: 'Memory and Context',
        content: `Effective multi-agent systems require robust memory and context management to maintain coherence across agent interactions. This is essential for agents to build upon each other's work and maintain a shared understanding of the task.

Key concepts of memory and context in multi-agent systems:

• Shared State
  - Central repository of information accessible to all agents
  - Provides consistent context across agent interactions
  - Can include conversation history, intermediate results, and task status
  - Enables agents to build upon each other's work

• Memory Types
  - Short-term Memory: Recent interactions and immediate context
  - Long-term Memory: Persistent knowledge and historical information
  - Working Memory: Task-specific information needed for current operations
  - Episodic Memory: Records of past agent interactions and outcomes

• Context Management
  - Selective Context: Providing relevant information to each agent
  - Context Compression: Summarizing lengthy histories to avoid token limits
  - Context Retrieval: Fetching relevant information when needed
  - Context Prioritization: Emphasizing the most important information

• Implementation Approaches
  - Message History: Maintaining a record of all agent interactions
  - Vector Stores: Retrieving relevant context based on semantic similarity
  - Structured State: Organizing information in a structured format
  - Summarization: Condensing lengthy contexts to fit token limits

Effective memory and context management is particularly important as the complexity of tasks and the number of agents increase.

Learn more about memory and context in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/low_level/#state).`,
        codeExamples: [{
          typescript: `import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, END } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { RunnableSequence } from "@langchain/core/runnables";

// Define a more complex state interface with structured memory
interface AgentMemory {
  // Conversation history
  messages: any[];

  // Structured knowledge base that agents can update
  knowledge_base: {
    facts: string[];
    conclusions: string[];
    open_questions: string[];
  };

  // Task tracking
  tasks: {
    completed: string[];
    pending: string[];
    current?: string;
  };

  // Current agent and completion status
  current_agent?: string;
  task_complete: boolean;
}

// Initialize the language model
const llm = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0
});

// Define agent roles
const agentRoles = {
  coordinator: "You are a coordination agent that manages tasks and knowledge. You organize information and delegate tasks to other agents.",
  researcher: "You are a research agent that investigates topics and adds facts to the knowledge base.",
  analyst: "You are an analytical agent that draws conclusions from facts in the knowledge base.",
  critic: "You are a critical thinking agent that identifies gaps in knowledge and adds open questions."
};

// Create a function to update the knowledge base
function updateKnowledgeBase(state: AgentMemory, updates: Partial<AgentMemory['knowledge_base']>) {
  return {
    ...state,
    knowledge_base: {
      ...state.knowledge_base,
      ...updates
    }
  };
}

// Create a function to update the task list
function updateTasks(state: AgentMemory, updates: Partial<AgentMemory['tasks']>) {
  return {
    ...state,
    tasks: {
      ...state.tasks,
      ...updates
    }
  };
}

// Create agent nodes with access to structured memory
function createAgentNode(role: string) {
  return async (state: AgentMemory) => {
    // Create a prompt for this agent that includes structured memory
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", agentRoles[role]],
      ["human", "Main task: {main_task}"],
      ["human", "Knowledge Base:\n- Facts: {facts}\n- Conclusions: {conclusions}\n- Open Questions: {open_questions}"],
      ["human", "Tasks:\n- Completed: {completed_tasks}\n- Pending: {pending_tasks}\n- Current: {current_task}"],
      ["human", "Recent conversation history: {history}"],
      ["human", "Based on the above information, perform your role as the {role} agent. Update the knowledge base and task list as appropriate."]
    ]);

    // Create a chain for this agent
    const chain = RunnableSequence.from([
      prompt,
      llm,
      new StringOutputParser()
    ]);

    // Get the main task from the first message
    const mainTask = state.messages[0].content;

    // Format the structured memory for the prompt
    const facts = state.knowledge_base.facts.map(f => '  * ' + f).join("\n") || "None yet";
    const conclusions = state.knowledge_base.conclusions.map(c => '  * ' + c).join("\n") || "None yet";
    const openQuestions = state.knowledge_base.open_questions.map(q => '  * ' + q).join("\n") || "None yet";

    const completedTasks = state.tasks.completed.map(t => '  * ' + t).join("\n") || "None yet";
    const pendingTasks = state.tasks.pending.map(t => '  * ' + t).join("\n") || "None yet";
    const currentTask = state.tasks.current || "None assigned";

    // Format recent conversation history (last 3 messages)
    const recentMessages = state.messages.slice(-3);
    const history = recentMessages.map(m =>
      (m.name || 'user') + ': ' + m.content
    ).join("\n");

    // Invoke the chain
    const result = await chain.invoke({
      main_task: mainTask,
      facts,
      conclusions,
      open_questions: openQuestions,
      completed_tasks: completedTasks,
      pending_tasks: pendingTasks,
      current_task: currentTask,
      history,
      role
    });

    // Parse the result to extract updates to the knowledge base and tasks
    // This is a simplified parsing approach - in a real system, you would use a more robust method
    const newFacts = extractListItems(result, "NEW FACTS:", "END FACTS");
    const newConclusions = extractListItems(result, "NEW CONCLUSIONS:", "END CONCLUSIONS");
    const newQuestions = extractListItems(result, "NEW QUESTIONS:", "END QUESTIONS");
    const newTasks = extractListItems(result, "NEW TASKS:", "END TASKS");
    const completedTaskMarkers = extractListItems(result, "COMPLETED TASKS:", "END COMPLETED");

    // Update the knowledge base with new information
    const updatedKnowledgeBase = {
      facts: [...state.knowledge_base.facts, ...newFacts],
      conclusions: [...state.knowledge_base.conclusions, ...newConclusions],
      open_questions: [...state.knowledge_base.open_questions, ...newQuestions]
    };

    // Update the task list
    const remainingPendingTasks = state.tasks.pending.filter(
      task => !completedTaskMarkers.some(marker => task.includes(marker))
    );

    const updatedTasks = {
      completed: [...state.tasks.completed, ...completedTaskMarkers],
      pending: [...remainingPendingTasks, ...newTasks],
      current: state.tasks.current
    };

    // Check if the main task is complete
    const taskComplete = result.toLowerCase().includes("main task complete");

    // Return the updated state
    return {
      messages: [
        ...state.messages,
        new AIMessage({
          content: result,
          name: role
        })
      ],
      knowledge_base: updatedKnowledgeBase,
      tasks: updatedTasks,
      task_complete: taskComplete
    };
  };
}

// Helper function to extract list items from text
function extractListItems(text: string, startMarker: string, endMarker: string): string[] {
  const startIndex = text.indexOf(startMarker);
  if (startIndex === -1) return [];

  const endIndex = text.indexOf(endMarker, startIndex);
  if (endIndex === -1) return [];

  const listSection = text.substring(startIndex + startMarker.length, endIndex).trim();
  return listSection.split("\n")
    .map(line => line.trim())
    .filter(line => line.startsWith("-") || line.startsWith("*"))
    .map(line => line.substring(1).trim())
    .filter(line => line.length > 0);
}

// Create the graph
const workflow = new StateGraph<AgentMemory>({
  channels: {
    messages: {
      value: (x: any[]) => x,
      default: () => []
    },
    knowledge_base: {
      value: (x: AgentMemory['knowledge_base']) => x,
      default: () => ({
        facts: [],
        conclusions: [],
        open_questions: []
      })
    },
    tasks: {
      value: (x: AgentMemory['tasks']) => x,
      default: () => ({
        completed: [],
        pending: [],
        current: undefined
      })
    },
    current_agent: {
      value: (x: string) => x,
      default: () => undefined
    },
    task_complete: {
      value: (x: boolean) => x,
      default: () => false
    }
  }
});

// Add nodes to the graph
workflow.addNode("coordinator", createAgentNode("coordinator"));
workflow.addNode("researcher", createAgentNode("researcher"));
workflow.addNode("analyst", createAgentNode("analyst"));
workflow.addNode("critic", createAgentNode("critic"));

// Define the router node
async function routerNode(state: AgentMemory) {
  if (state.task_complete) {
    return { ...state, current_agent: "end" };
  }

  // Simple routing logic based on the current state
  if (state.knowledge_base.facts.length < 3) {
    // Need more facts, use the researcher
    return { ...state, current_agent: "researcher" };
  } else if (state.knowledge_base.conclusions.length < 2) {
    // Need more analysis, use the analyst
    return { ...state, current_agent: "analyst" };
  } else if (state.knowledge_base.open_questions.length < 2) {
    // Need more critical thinking, use the critic
    return { ...state, current_agent: "critic" };
  } else {
    // Default to coordinator for task management
    return { ...state, current_agent: "coordinator" };
  }
}

workflow.addNode("router", routerNode);

// Define the edges
workflow.setEntryPoint("router");

// Add conditional edges from the router
workflow.addConditionalEdges(
  "router",
  (state) => state.current_agent
);

// Add edges from each agent back to the router
workflow.addEdge("coordinator", "router");
workflow.addEdge("researcher", "router");
workflow.addEdge("analyst", "router");
workflow.addEdge("critic", "router");

// Compile the graph
const memoryAwareSystem = workflow.compile();

// Example usage
async function runMemoryAwareSystem() {
  const result = await memoryAwareSystem.invoke({
    messages: [
      new HumanMessage("Research the impact of climate change on agriculture and provide recommendations for sustainable farming practices.")
    ],
    knowledge_base: {
      facts: [],
      conclusions: [],
      open_questions: []
    },
    tasks: {
      completed: [],
      pending: ["Research climate change impacts", "Identify sustainable practices", "Formulate recommendations"],
      current: "Research climate change impacts"
    },
    task_complete: false
  });

  console.log("Final knowledge base:");
  console.log("Facts:", result.knowledge_base.facts);
  console.log("Conclusions:", result.knowledge_base.conclusions);
  console.log("Open Questions:", result.knowledge_base.open_questions);

  console.log("\nTask status:");
  console.log("Completed:", result.tasks.completed);
  console.log("Pending:", result.tasks.pending);
}

runMemoryAwareSystem();`,
          python: `from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableSequence
from typing import TypedDict, List, Dict, Any, Optional
import re

# Define a more complex state interface with structured memory
class KnowledgeBase(TypedDict):
    facts: List[str]
    conclusions: List[str]
    open_questions: List[str]

class Tasks(TypedDict):
    completed: List[str]
    pending: List[str]
    current: Optional[str]

class AgentMemory(TypedDict):
    # Conversation history
    messages: List[Any]

    # Structured knowledge base that agents can update
    knowledge_base: KnowledgeBase

    # Task tracking
    tasks: Tasks

    # Current agent and completion status
    current_agent: Optional[str]
    task_complete: bool

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Define agent roles
agent_roles = {
    "coordinator": "You are a coordination agent that manages tasks and knowledge. You organize information and delegate tasks to other agents.",
    "researcher": "You are a research agent that investigates topics and adds facts to the knowledge base.",
    "analyst": "You are an analytical agent that draws conclusions from facts in the knowledge base.",
    "critic": "You are a critical thinking agent that identifies gaps in knowledge and adds open questions."
}

# Helper function to extract list items from text
def extract_list_items(text: str, start_marker: str, end_marker: str) -> List[str]:
    pattern = f"{start_marker}(.*?){end_marker}"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []

    list_section = match.group(1).strip()
    items = []
    for line in list_section.split("\n"):
        line = line.strip()
        if line.startswith("-") or line.startswith("*"):
            item = line[1:].strip()
            if item:
                items.append(item)

    return items

# Create agent nodes with access to structured memory
def create_agent_node(role: str):
    async def agent_node(state: AgentMemory) -> AgentMemory:
        # Create a prompt for this agent that includes structured memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_roles[role]),
            ("human", "Main task: {main_task}"),
            ("human", "Knowledge Base:\n- Facts: {facts}\n- Conclusions: {conclusions}\n- Open Questions: {open_questions}"),
            ("human", "Tasks:\n- Completed: {completed_tasks}\n- Pending: {pending_tasks}\n- Current: {current_task}"),
            ("human", "Recent conversation history: {history}"),
            ("human", "Based on the above information, perform your role as the {role} agent. Update the knowledge base and task list as appropriate.")
        ])

        # Create a chain for this agent
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        # Get the main task from the first message
        main_task = state["messages"][0].content

        # Format the structured memory for the prompt
        facts = "\n".join([f"  * {f}" for f in state["knowledge_base"]["facts"]]) or "None yet"
        conclusions = "\n".join([f"  * {c}" for c in state["knowledge_base"]["conclusions"]]) or "None yet"
        open_questions = "\n".join([f"  * {q}" for q in state["knowledge_base"]["open_questions"]]) or "None yet"

        completed_tasks = "\n".join([f"  * {t}" for t in state["tasks"]["completed"]]) or "None yet"
        pending_tasks = "\n".join([f"  * {t}" for t in state["tasks"]["pending"]]) or "None yet"
        current_task = state["tasks"]["current"] or "None assigned"

        # Format recent conversation history (last 3 messages)
        recent_messages = state["messages"][-3:] if len(state["messages"]) > 3 else state["messages"]
        history = "\n".join([
            f"{m.name or 'user'}: {m.content}"
            for m in recent_messages
        ])

        # Invoke the chain
        result = await chain.ainvoke({
            "main_task": main_task,
            "facts": facts,
            "conclusions": conclusions,
            "open_questions": open_questions,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "current_task": current_task,
            "history": history,
            "role": role
        })

        # Parse the result to extract updates to the knowledge base and tasks
        new_facts = extract_list_items(result, "NEW FACTS:", "END FACTS")
        new_conclusions = extract_list_items(result, "NEW CONCLUSIONS:", "END CONCLUSIONS")
        new_questions = extract_list_items(result, "NEW QUESTIONS:", "END QUESTIONS")
        new_tasks = extract_list_items(result, "NEW TASKS:", "END TASKS")
        completed_task_markers = extract_list_items(result, "COMPLETED TASKS:", "END COMPLETED")

        # Update the knowledge base with new information
        updated_knowledge_base = {
            "facts": state["knowledge_base"]["facts"] + new_facts,
            "conclusions": state["knowledge_base"]["conclusions"] + new_conclusions,
            "open_questions": state["knowledge_base"]["open_questions"] + new_questions
        }

        # Update the task list
        remaining_pending_tasks = [
            task for task in state["tasks"]["pending"]
            if not any(marker in task for marker in completed_task_markers)
        ]

        updated_tasks = {
            "completed": state["tasks"]["completed"] + completed_task_markers,
            "pending": remaining_pending_tasks + new_tasks,
            "current": state["tasks"]["current"]
        }

        # Check if the main task is complete
        task_complete = "main task complete" in result.lower()

        # Return the updated state
        return {
            "messages": [
                *state["messages"],
                AIMessage(content=result, name=role)
            ],
            "knowledge_base": updated_knowledge_base,
            "tasks": updated_tasks,
            "task_complete": task_complete
        }

    return agent_node

# Create the graph
workflow = StateGraph(AgentMemory)

# Add nodes to the graph
workflow.add_node("coordinator", create_agent_node("coordinator"))
workflow.add_node("researcher", create_agent_node("researcher"))
workflow.add_node("analyst", create_agent_node("analyst"))
workflow.add_node("critic", create_agent_node("critic"))

# Define the router node
async def router_node(state: AgentMemory) -> AgentMemory:
    if state["task_complete"]:
        return {**state, "current_agent": "end"}

    # Simple routing logic based on the current state
    if len(state["knowledge_base"]["facts"]) < 3:
        # Need more facts, use the researcher
        return {**state, "current_agent": "researcher"}
    elif len(state["knowledge_base"]["conclusions"]) < 2:
        # Need more analysis, use the analyst
        return {**state, "current_agent": "analyst"}
    elif len(state["knowledge_base"]["open_questions"]) < 2:
        # Need more critical thinking, use the critic
        return {**state, "current_agent": "critic"}
    else:
        # Default to coordinator for task management
        return {**state, "current_agent": "coordinator"}

workflow.add_node("router", router_node)

# Define the edges
workflow.set_entry_point("router")

# Add conditional edges from the router
workflow.add_conditional_edges(
    "router",
    lambda state: state["current_agent"]
)

# Add edges from each agent back to the router
workflow.add_edge("coordinator", "router")
workflow.add_edge("researcher", "router")
workflow.add_edge("analyst", "router")
workflow.add_edge("critic", "router")

# Compile the graph
memory_aware_system = workflow.compile()

# Example usage
async def run_memory_aware_system():
    result = await memory_aware_system.ainvoke({
        "messages": [
            HumanMessage(content="Research the impact of climate change on agriculture and provide recommendations for sustainable farming practices.")
        ],
        "knowledge_base": {
            "facts": [],
            "conclusions": [],
            "open_questions": []
        },
        "tasks": {
            "completed": [],
            "pending": ["Research climate change impacts", "Identify sustainable practices", "Formulate recommendations"],
            "current": "Research climate change impacts"
        },
        "task_complete": False
    })

    print("Final knowledge base:")
    print("Facts:", result["knowledge_base"]["facts"])
    print("Conclusions:", result["knowledge_base"]["conclusions"])
    print("Open Questions:", result["knowledge_base"]["open_questions"])

    print("\nTask status:")
    print("Completed:", result["tasks"]["completed"])
    print("Pending:", result["tasks"]["pending"])

import asyncio
asyncio.run(run_memory_aware_system())`,
          description: 'Implementing a multi-agent system with structured memory and knowledge management'
        }]
      }
    ]
  }
};
