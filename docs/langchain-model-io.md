# Langchain Model I/O Concepts

## LLM and ChatModels

Language models in LangChain come in two flavors:

* ChatModel: models that are trained to have conversations, including GPT-4 and Claude-2
* LLM in LangChain refer to pure text completion models, like GPT-3.
* Different models have different prompting strategies that work best for them. **For example, Anthropic's models work best with XML while OpenAI's work best with JSON.**

```python
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

llm = OpenAI()
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

The main difference between them is their input and output schemas. The LLM objects take string as input and output string. The ChatModel objects take a list of messages as input and output a message.

```python
from langchain_core.messages import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

llm.invoke(text)
# >> Feetful of Fun

chat_model.invoke(messages)
# >> AIMessage(content="Socks O'Color")
```

## Prompt Templates

Most LLM applications do not pass user input directly into an LLM. Usually they will add the user input to a larger piece of text, called a prompt template, that provides additional context on the specific task at hand.

PromptTemplates help bundle up all the logic for going from user input into a fully formatted prompt. 

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt.format(product="colorful socks")
```

`PromptTemplate`s can also be used to produce a list of messages. In this case, the prompt not only contains information about the content, but also each message (its role, its position in the list, etc.). Here, what happens most often is a `ChatPromptTemplate` is a list of `ChatMessageTemplates`. Each `ChatMessageTemplate` contains instructions for how to format that `ChatMessage` - its role, and then also its content.

```python
from langchain_core.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
```

* You can provide values of the variables later together for multiple templates.

## Output parsers

There are different parsers to parse the model outputs. Types of parsers includes

- Convert text from `LLM` into structured information (e.g. JSON), `SimpleJsonOutputParser`
- Convert a `ChatMessage` into just a string, `StrOutputParser`
- Convert the extra information returned from a call besides the message (like OpenAI function invocation) into a string.

An example that parses a list of comma separated values.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
output_parser.parse("hi, bye")
# >> ['hi', 'bye']
```

## Composing with LCEL

We can now combine all these into one chain.

```python
template = "Generate a list of 5 {text}.\n\n{format_instructions}"

chat_prompt = ChatPromptTemplate.from_template(template)
chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
chain = chat_prompt | chat_model | output_parser
chain.invoke({"text": "colors"})
# >> ['red', 'blue', 'green', 'yellow', 'orange']
```

Note that we are using the `|` syntax to join these components together. This `|` syntax is powered by the LangChain Expression Language (LCEL) and relies on the universal `Runnable` interface that all of these objects implement.

























