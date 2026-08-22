from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import sys
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.llm.graph import ask

client = Client()

class ab_bot_eval:
    def __init__(self):
        self.dataset_name="a-b dataset"

    def start_eval(self):
        result=client.evaluate(
            self.ab_app,
            data=self.dataset_name,
            evaluators=[self.correctness],
            experiment_prefix="vasu-robin"
            )
        print("result---", result)
        failures = []
        print("failures---", failures)
        for row in result:
            for eval_result in row["evaluation_results"]["results"]:
                if eval_result.key == "correctness" and eval_result.score is not True:
                    failures.append(row["example"].id)

        assert not failures, f"Correctness failed for examples: {failures}"
        print("All examples passed correctness")
        return None

    @staticmethod
    def correctness(
            inputs: dict,
            outputs: dict,
            reference_outputs: dict
            )->bool:
        eval_instructions = """
        You are an expert evaluator. Compare the assistant's answer with the reference answer.
        Return True if the answer is substantially correct and preserves the main conclusions,
        allowing differences in wording, structure, and minor omissions. Return False only if
        there is a major factual error, contradiction, or different conclusion.
        """
        user_content = f"""You are grading the following question:
        {inputs['question']}
        Here is the real answer:
        {reference_outputs['answer']}
        You are grading the following predicted answer:
        {outputs['response']}
        Respond with CORRECT or INCORRECT:
        Grade:"""
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=1)

        response = llm.invoke([
            SystemMessage(content=eval_instructions),
            HumanMessage(content=user_content)
        ]).content.strip().upper()
        return "CORRECT" in response and "INCORRECT" not in response

    @staticmethod
    def ab_app(
            inputs: dict
            ):
         thread_id = str(uuid.uuid4())
         BASE_DIR = os.path.dirname(os.path.abspath(__file__))
         file_path = os.path.join(BASE_DIR, "marketing_AB.csv")
         ans_ = ask(user_message=inputs["question"], user_id="139fd288-bbfd-4e9b-9729-b7d8f6299b10", thread_id=thread_id, csv_path=file_path)
         return {"response": ans_}