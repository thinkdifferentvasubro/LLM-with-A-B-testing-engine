import os
from langchain_google_genai import ChatGoogleGenerativeAI
from tool import run_ab_test_analysis, generate_csv_schema, rag_search
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated, NotRequired
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import trim_messages
import uuid


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ab_prompt_PATH = os.path.join(BASE_DIR, "a-b_prompt.txt")
rag_prompt_path = os.path.join(BASE_DIR, "rag_prompt.txt")

with open(ab_prompt_PATH, "r") as a_f:
    ab_prompt = a_f.read()
with open(rag_prompt_path, "r") as r_f:
    rag_prompt = r_f.read()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    csv_path: str | None
    user_id: str

    message_intent: NotRequired[str]
    ab_test_result: NotRequired[str]
    rag_result: NotRequired[str]

class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "retrieve tests", "conduct test", "both"]

class test_params(BaseModel):
    episodes: list[dict]
    Covariate_cols: list[str]
    cat_cols: list[str]
    num_cols: list[str]
    date_and_formats: dict
    num_col_with_str_vals: dict

class rag_params(BaseModel):
    query: str
    n_results: int

def classify_intent(state: State):
    structured_lmm = llm.with_structured_output(IntentClassifier)
    result = structured_lmm.invoke([
        {
            "role": "system",
            "content": """Classify the user's message as one of: "conduct test" (run a new A/B test or comparison), "retrieve tests" (retrieve previously stored A/B test or covariate balance results), "both" (retrieve previous results and run a new test), or "chat" (general conversation, greetings, questions, or any request that does not require running or retrieving A/B tests). Return only one label."""
        },
        {
            "role": "user",
            "content": state["messages"][-1].content
        }
    ])
    return {"message_intent": result.message_intent}

def conduct_test(state: State):
    if not state.get("csv_path"):
        return {"ab_test_result": "ask the user to upload the csv"}

    else:
        schema, dataset = generate_csv_schema(csv_path=state["csv_path"])
        formatted_ab_prompt = ab_prompt.format(schema=schema)
        structured_llm = llm.with_structured_output(test_params)
        trimmed = trim_messages(
            state["messages"],
            max_tokens=2900,
            strategy="last",
            token_counter=llm,
            include_system=False,
            start_on="human"
        )
        params = structured_llm.invoke(
            [
                {
                    "role": "system",
                    "content": formatted_ab_prompt
                }
            ]+trimmed
        )
        test_result = run_ab_test_analysis(
            **params.model_dump(),
            dataset=dataset,
            user_id = state["user_id"]
        )
        return {"ab_test_result": test_result}

def rag_retrieval(state: State):
    structured_llm = llm.with_structured_output(rag_params)
    trimmed = trim_messages(
        state["messages"],
        max_tokens=2900,
        strategy="last",
        token_counter=llm,
        include_system=False,
        start_on="human"
    )
    params = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": rag_prompt
            }
        ]+trimmed
    )
    rag_result = rag_search(**params.model_dump(), user_id=state["user_id"])
    return {"rag_result": rag_result}

def both_(state: State):
    results = {}
    r = rag_retrieval(state)
    c = conduct_test(state)
    results.update(r)
    results.update(c)
    return results

def reasoning(state: State):
    tool_messages = []

    if state.get("ab_test_result"):
        tool_messages.append({
            "role": "assistant",
            "content": f"[Tool: A/B test analysis]\n{state['ab_test_result']}"
        })

    if state.get("rag_result"):
        tool_messages.append({
            "role": "assistant",
            "content": f"[Tool: Historical/RAG retrieval]\n{state['rag_result']}"
        })

    system_content = """You are an A/B testing assistant. Respond naturally to general conversation, and explain any provided A/B test results, retrieved historical results, or both in clear, concise language. Interpret the statistical test, p-value, effect direction, practical significance, and any covariate balance findings. If covariates you selected are flagged as imbalanced, acknowledge that they may not be suitable for adjustment and explain why. If the imbalance comes from user-specified covariates, politely inform the user that those covariates may introduce confounding and suggest choosing more balanced pre-treatment covariates. When both current and historical results are available, compare them, highlight consistent or conflicting findings, and answer using only the provided information without inventing facts. If the user asks to save an A/B test or its results, explain that all executed tests are automatically stored in the database and can be retrieved later, so no manual save operation is required. Tool outputs will appear in the conversation as assistant messages prefixed with "[Tool: ...]". Treat these as ground-truth results to reason over, not as your own prior statements."""

    trimmed = trim_messages(
        state["messages"] + tool_messages,
        max_tokens=2900,
        strategy="last",
        token_counter=llm,
        include_system=False,
        start_on="human"
        )
    response = llm.invoke(
        [
            {"role": "system", "content": system_content}
        ] + trimmed
    )

    return {"messages": tool_messages + [{"role": "assistant", "content": response.content}]}

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