# Function Calling

Many LLM providers, including Anthropic OpenAI, and others, support variants of a tool calling feature. These features typically allow requests to the LLM to include available tools and their schemas, and for responses to include calls to these tools. For instance, given a search engine tool, an LLM might handle a query by first issuing a call to the search engine. The system calling the LLM can receive the tool call, execute it, and return the output to the LLM to inform its response. 

OpenAI separates tool calls into a distinct parameter, with arguments as JSON strings:

```json
{
  "tool_calls": [
    {
      "id": "id_value",
      "function": {
        "arguments": '{"arg_name": "arg_value"}',
        "name": "tool_name"
      },
      "type": "function"
    }
  ]
}
```

LangChain implements standard interfaces for defining tools, passing them to LLMs, and representing tool calls.

## Request: Passing tools to model

For a model to be able to invoke tools, you need to pass tool schemas to it when making a chat request. LangChain ChatModels supporting tool calling features implement a `.bind_tools` method.

### Defining tool schemas: LangChain Tool

For example, we can define the schema for custom tools using the `@tool` decorator on Python functions:

```python
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


tools = [add, multiply]
```

### Defining tool schemas: Pydantic class

We can equivalently define the schema using Pydantic. Pydantic is useful when your tool inputs are more complex:

```python
from langchain_core.pydantic_v1 import BaseModel, Field


# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [add, multiply]
```

### Binding tool schemas

We can use the `bind_tools()` method to handle converting `Multiply` to a "tool" and binding it to the model (i.e., passing it in each time the model is invoked).

```python
llm_with_tools = llm.bind_tools(tools)
```

## Request: Passing tool outputs to model

If we're using the model-generated tool invocations to actually call tools and want to pass the tool results back to the model, we can do so using `ToolMessage`s.

```python
from langchain_core.messages import HumanMessage, ToolMessage


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

messages
```

