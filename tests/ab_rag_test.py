import os
import sys
import uuid
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.RAG.rag_pipeline import VectorDBManager

client = Client()

class rag_eval:
    def __init__(self):
        self.dataset_name="rag Evaluation"

    def start_rag_eval(self):
        result = client.evaluate(
            self.rag_search,
            data=self.dataset_name,
            evaluators=[self.rag_correctness],
            experiment_prefix="vasu-robin"
            )

        failures = []
        for row in result:
            for eval_result in row["evaluation_results"]["results"]:
                if eval_result.key == "rag_correctness" and eval_result.score is not True:
                    failures.append(row["example"].id)

        assert not failures, f"RAG correctness failed for examples: {failures}"
        print("All examples passed rag_correctness ✅")
        return None

    @staticmethod
    def rag_correctness(
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
    def rag_search(
        inputs: dict
        ) -> dict:
    
        user_id = "139fd288-bbfd-4e9b-9729-b7d8f6299b10"
        n_results = 1
        rag_results = VectorDBManager().retrieve_data(
            query=inputs["question"],
            n_results=n_results,
            user_id=user_id
            )
        
        final_rag_result = "Rag results: "
        for rag_result in rag_results["documents"]:
            if not rag_result:
                return "could not find the related test/experiment"
            final_rag_result += rag_result[0]
        return {"response": final_rag_result}
    