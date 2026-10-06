# Understanding and dry run of `conditional_chain.py`

This document explains the current file without changing it or calling the model. Model responses below are illustrative: actual classification and reply text can vary.

## 1. What the program does

The program classifies customer feedback as positive or negative, selects the matching response chain, generates a short reply, and prints it.

```text
Input dictionary
    → classification prompt
    → model call 1
    → Pydantic Feedback object
    → branch condition
    → positive or negative reply prompt
    → model call 2
    → plain string
    → print(result)
```

**Important detail in the current code:** the original customer feedback does not reach the reply prompt. The classifier returns only a sentiment object. The branch receives that object and passes it to the reply prompt. This means the reply is based on `sentiment='positive'` or `sentiment='negative'`, rather than the customer's actual words. The dry run below shows this behavior explicitly.

## 2. Imports and setup

| Import | Purpose in this file |
|---|---|
| `load_dotenv` | Loads variables from a `.env` file into the environment, such as the API key. |
| `ChatOpenAI` | Creates the chat model used for both classification and reply generation. |
| `PromptTemplate` | Inserts values into prompt text. |
| `StrOutputParser` | Converts the model's reply into a plain Python string. |
| `PydanticOutputParser` | Parses the classification response into the specified Pydantic model. |
| `RunnableBranch` | Evaluates conditions in order and runs the first matching branch. |
| `RunnableLambda` | Wraps the fallback Python function as a runnable. |
| `BaseModel`, `Field` | Define a validated data model and describe its field. |
| `Literal` | Restricts the sentiment value to the two allowed strings. |

```python
load_dotenv()
model = ChatOpenAI(model='gpt-5-nano-2025-08-07')
parser1 = StrOutputParser()
```

These statements prepare the environment, model, and text parser. Constructing the model does not generate a reply. Model calls happen when the chain is invoked.

## 3. The `Feedback` model

```python
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )
```

This class defines the expected classification result. It has one field, `sentiment`, whose valid values are exactly `"positive"` and `"negative"`.

```python
Feedback(sentiment="positive")  # Valid object
Feedback(sentiment="negative")  # Valid object
Feedback(sentiment="neutral")   # Validation error
```

The description helps explain the field in the schema. `Literal` supplies the allowed-value restriction.

```python
parser2 = PydanticOutputParser(pydantic_object=Feedback)
```

`parser2` expects model output that can be parsed and validated as a `Feedback` object. It does not itself classify the customer's text; the model performs classification.

## 4. The classification prompt and chain

```python
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2
```

- `{feedback}` is supplied when the chain runs.
- `{format_instructions}` is already supplied through `partial_variables`. The parser generates instructions describing the JSON output and the `Feedback` schema.
- The `|` operator connects runnable stages: the output of one becomes the input of the next.
- Creating this chain defines the execution sequence; it does not execute it.

The word `postive` is a spelling mistake in the current prompt. The allowed schema values remain `positive` and `negative`.

## 5. The two reply prompts

`prompt2` asks for a positive-feedback reply. `prompt3` asks for a negative-feedback reply. Both request 1–2 sentences and only the message, with no options, tips, headings, placeholders, or questions.

Both templates have exactly one input variable:

```python
input_variables=["feedback"]
```

This matters because the installed LangChain prompt implementation accepts a non-dictionary input for a single-variable prompt. It puts that input into the sole variable. In this file, that input is a `Feedback` object, so the object itself becomes the value of `{feedback}`.

The instruction to return only a short message guides the model; it is not a programmatic length or formatting validator.

## 6. Branch selection

```python
conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser1),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)
```

Here, `x` is the output of `classifier_chain`: a `Feedback` object.

The branch checks the positive condition first. If it is true, it runs the positive reply chain and stops checking conditions. Otherwise, it checks the negative condition. If neither condition matches, it runs the final fallback runnable.

The matching reply chain receives the same input object that the branch received. The condition returns a Boolean; that Boolean is used to select the branch and does not become the reply prompt's input.

### Why `x.sentiment` works

```python
x = Feedback(sentiment="positive")
x.sentiment       # "positive"
x["sentiment"]   # TypeError: 'Feedback' object is not subscriptable
```

Pydantic models normally use attribute access. Dictionary access requires conversion first:

```python
data = x.model_dump()
data["sentiment"]  # "positive"
```

The current classifier returns a model object, so `x.sentiment` is correct in the current branch conditions.

## 7. Full dry run for the current input

```python
chain = classifier_chain | conditional_chain
result = chain.invoke({"feedback": "this smartphone works well"})
```

### Step 1: Supply the input

The input is a Python dictionary:

```python
{"feedback": "this smartphone works well"}
```

It enters `classifier_chain` first.

### Step 2: Format `prompt1`

The prompt receives the original feedback and adds the parser's format instructions. Its content is approximately:

```text
Classify the sentiment of the following feedback text into postive or negative
this smartphone works well
[Instructions to return JSON matching the Feedback schema]
```

The bracketed line summarizes the actual generated instructions; it is not their exact text. A `PromptTemplate` invocation returns a prompt value for the model.

### Step 3: First model call

The model reads the classification prompt. An expected response is an `AIMessage` containing:

```json
{"sentiment": "positive"}
```

This is an illustrative response, not a recorded API result.

### Step 4: Parse and validate the classification

`parser2` extracts and parses the response, validates it against `Feedback`, and returns:

```python
Feedback(sentiment="positive")
```

Its type is `Feedback`, not `dict` or `str`. The object contains no field holding `"this smartphone works well"`.

### Step 5: Evaluate the first branch condition

```python
x = Feedback(sentiment="positive")
x.sentiment == "positive"  # True
```

The positive branch is selected:

```python
prompt2 | model | parser1
```

The negative condition and fallback do not run.

### Step 6: Format the positive reply prompt

`prompt2` receives the `Feedback` object. Because it has only one variable, LangChain treats that object as the value for `feedback`, conceptually:

```python
{"feedback": Feedback(sentiment="positive")}
```

Formatting converts the object to text. The model sees approximately:

```text
Write one short, ready-to-send reply to this customer's positive feedback
in 1-2 sentences. Output only the message—no options, tips, headings,
placeholders, or questions.
sentiment='positive'
```

**It does not see `this smartphone works well` at this stage.** The original input was replaced by the classifier's output in the sequential chain.

### Step 7: Second model call

The model generates a reply based on the positive-response instructions and the sentiment text. For example, its `AIMessage` might contain:

```text
Thank you for your kind feedback! We're glad you had a positive experience.
```

Since the smartphone feedback is absent, a generic reply is a reasonable outcome. Exact wording is not guaranteed.

### Step 8: Parse the reply as a string

`parser1` converts the message into a plain Python string:

```python
"Thank you for your kind feedback! We're glad you had a positive experience."
```

### Step 9: Assign and print the result

`chain.invoke(...)` returns this final string, which is assigned to `result`.

```python
print(result)
```

The console prints the reply. It does not automatically print the classifier result or intermediate prompts.

## 8. Types at each stage

| Stage | Input | Output |
|---|---|---|
| `prompt1` | Dictionary containing original feedback | Prompt value |
| First `model` invocation | Classification prompt | `AIMessage` |
| `parser2` | Classification message | `Feedback` object |
| Branch condition | `Feedback` object | Boolean used for selection |
| Selected `prompt2` or `prompt3` | The same `Feedback` object | Reply prompt value |
| Second `model` invocation | Reply prompt | `AIMessage` |
| `parser1` | Reply message | `str` |
| `chain.invoke(...)` | Initial dictionary | Final reply string |

## 9. Negative-feedback dry run

For an input such as:

```python
{"feedback": "this smartphone keeps crashing"}
```

Assuming the model classifies it correctly:

1. `prompt1` includes the original complaint.
2. The first model call returns JSON containing `"sentiment": "negative"`.
3. `parser2` returns `Feedback(sentiment="negative")`.
4. The positive condition returns `False`.
5. The negative condition returns `True`.
6. `prompt3` receives that object and inserts `sentiment='negative'` into `{feedback}`.
7. The second model call produces a short negative-feedback reply.
8. `parser1` returns the reply string, which is printed.

The reply prompt still does not receive the actual complaint about crashes.

## 10. Fallback and errors

```python
RunnableLambda(lambda x: "could not find sentiment")
```

This fallback runs only when neither condition matches. It does not catch exceptions.

With the current validated `Feedback` schema, a successfully parsed sentiment must be positive or negative, so one of the two conditions should match. If the model returns an invalid value such as `neutral`, parsing normally raises an error before the branch is reached. Invalid JSON, API failures, or missing credentials can also interrupt execution; the fallback does not handle these failures.

A normal successful invocation makes two model calls: one for classification and one for the selected reply. It does not call both reply branches.

## 11. Commented-out code

The lines starting with `#` do not execute. In the current file, this includes all three graph-printing calls and the extra classifier example at the bottom.

```python
# chain.get_graph().print_ascii()
# classifier_chain.get_graph().print_ascii()
# conditional_chain.get_graph().print_ascii()
```

If enabled, these display the runnable structure. Drawing the graph does not invoke the language model. ASCII graph drawing requires `grandalf`, which was installed earlier in this session.

## 12. Main behavior to remember

`classifier_chain | conditional_chain` passes the classifier's output to the branch. It does not automatically preserve the initial input dictionary.

For a reply that refers to the actual customer's feedback, the data passed to the branch would need to contain both the original feedback and its classification. That is a separate code change; the Python file has been left untouched for this documentation task.
