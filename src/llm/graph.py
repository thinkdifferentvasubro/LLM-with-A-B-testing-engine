import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.llm.tool import run_ab_test_analysis, generate_csv_schema, rag_search
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated, NotRequired, Union, Literal
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.messages import trim_messages


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
intent_prompt_PATH = os.path.join(BASE_DIR, "intent_prompt.txt")
ab_prompt_PATH = os.path.join(BASE_DIR, "a-b_prompt.txt")
rag_prompt_path = os.path.join(BASE_DIR, "rag_prompt.txt")
reasoning_prompt_PATH = os.path.join(BASE_DIR, "reasoning_prompt.txt")

with open(ab_prompt_PATH, "r") as a_f:
    ab_prompt = a_f.read()
with open(rag_prompt_path, "r") as r_f:
    rag_prompt = r_f.read()
with open(intent_prompt_PATH, "r") as i_f:
    intent_prompt = i_f.read()
with open(reasoning_prompt_PATH, "r") as re_f:
    reasoning_prompt = re_f.read()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, max_retries=1)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    csv_path: str | None
    user_id: str

    message_intent: NotRequired[str]
    ab_test_result: NotRequired[str]
    rag_result: NotRequired[str]

class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "retrieve tests", "conduct test", "both"]

Scalar = Union[bool, int, float, str]

class PairSpec(BaseModel):
    control_value: Scalar
    treatment: list[list[Scalar]] = Field(
        description='List of [column, value] pairs, e.g. [["test group", "ad"]]. '
                    'Multiple pairs allowed, e.g. [["test group", "ad"], ["region", "west"]]'
    )


class Episode(BaseModel):
    pairs: dict[str, PairSpec]
    metrics: list[str]
    p_value: float = 0.05
    test: Literal[
        "", "ttest", "welchttest", "mannwhitneyu", "anova",
        "kruskalwallis", "fisherexact", "ztest", "chisquare",
        "manova", "gtest", "welchanova"
    ] = ""
    tail: Literal["", "greater", "less"] = ""

class test_params(BaseModel):
    episodes: list[Episode]
    Covariate_cols: list[str]
    cat_cols: list[str]
    num_cols: list[str]
    date_and_formats: dict[str, str]
    num_col_with_str_vals: dict[str, str]

class rag_params(BaseModel):
    query: str
    n_results: int

def classify_intent(state: State):
    trimmed = trim_messages(
        state["messages"],
        max_tokens=1500,
        strategy="last",
        token_counter="approximate",
        include_system=False,
        start_on="human"
    )
    if not state.get("csv_path"):
        csv_status = " csv upload status: the csv is not uploaded by the user"
    else:
        csv_status = " csv upload status: the data csv is uploaded by the user"

    structured_lmm = llm.with_structured_output(IntentClassifier)
    result = structured_lmm.invoke([
        {
            "role": "system",
            "content": intent_prompt + csv_status
        }
    ] + trimmed)
    print("intent")
    return {
        "message_intent": result.message_intent,
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + trimmed,
    }

def conduct_test(state: State):
    if not state.get("csv_path"):
        print("no csv test")
        return {"ab_test_result": "The user did not upload his csv file ask him to upload the file for the test"}

    else:
        schema, dataset = generate_csv_schema(csv_path=state["csv_path"])
        if dataset is None:
            return {"ab_test_result": schema}
        formatted_ab_prompt = ab_prompt.format(schema=schema)
        structured_llm = llm.with_structured_output(test_params)
        params = structured_llm.invoke(
            [
                {
                    "role": "system",
                    "content": formatted_ab_prompt
                }
            ]+state["messages"]
        )
        print(params)
        test_result = run_ab_test_analysis(
            **params.model_dump(),
            dataset=dataset,
            user_id = state["user_id"]
        )
        print("test")
        return {"ab_test_result": test_result}

def rag_retrieval(state: State):
    structured_llm = llm.with_structured_output(rag_params)
    params = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": rag_prompt
            }
        ]+state["messages"]
    )
    rag_result = rag_search(**params.model_dump(), user_id=state["user_id"])
    print("rag")
    return {"rag_result": rag_result}

def both_(state: State):
    results = {}
    r = rag_retrieval(state)
    c = conduct_test(state)
    results.update(r)
    results.update(c)
    print("both")
    return results

def reasoning(state: State):
    tool_context = []

    if state.get("ab_test_result"):
        tool_context.append(
            f"[A/B TEST TOOL RESULT]\n{state['ab_test_result']}"
        )

    if state.get("rag_result"):
        tool_context.append(
            f"[HISTORICAL TEST TOOL RESULT]\n{state['rag_result']}"
        )

    context = "\n\n".join(tool_context)

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": reasoning_prompt
            },
            {
                "role": "user",
                "content": (
                    f"User's question:\n{state['messages'][-1].content}\n\n"
                    f"Tool results:\n{context}\n\n"
                    "Answer the user's question directly. "
                    "Interpret the results instead of repeating the tool output."
                )
            }
        ]
    )

    print("reasoning")

    return {
        "messages": [
            {
                "role": "assistant",
                "content": response.content
            }
        ]
    }
graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("ab_test_agent", conduct_test)
graph_builder.add_node("rag_agent", rag_retrieval)
graph_builder.add_node("reasoning", reasoning)
graph_builder.add_node("both_", both_)

graph_builder.add_edge(START, "classifier")

graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {
        "chat": "reasoning",
        "conduct test": "ab_test_agent",
        "retrieve tests": "rag_agent",
        "both": "both_"
    },
)

graph_builder.add_edge("ab_test_agent", "reasoning")
graph_builder.add_edge("rag_agent", "reasoning")
graph_builder.add_edge("both_", "reasoning")
graph_builder.add_edge("reasoning", END)

graph = graph_builder.compile(checkpointer=InMemorySaver())

def ask(
        user_id: str,
        user_message: str,
        thread_id: str,
        csv_path: str|None=None
        ):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
           ],
           "user_id": user_id,
           "csv_path": csv_path
        },
        config=config)
    return result["messages"][-1].content