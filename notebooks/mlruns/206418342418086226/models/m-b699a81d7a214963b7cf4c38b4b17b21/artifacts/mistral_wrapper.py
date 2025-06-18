import os
import mlflow
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

class MistralQA(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name="mistral-tiny", system_prompt="Answer the following question in two sentences", temperature=0.7, max_tokens=1000):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))

    def predict(self, context, model_input):
        questions = model_input.get("inputs", [])
        if not isinstance(questions, list):
            questions = [questions]
        responses = []
        for question in questions:
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=question)
            ]
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            responses.append(response.choices[0].message.content)
        return responses

mistral_qa_instance = MistralQA(
    model_name="mistral-tiny",
    system_prompt="Answer the following question in two sentences",
    temperature=0.7,
    max_tokens=1000
)

# Set the model for MLflow to discover
mlflow.models.set_model(mistral_qa_instance)