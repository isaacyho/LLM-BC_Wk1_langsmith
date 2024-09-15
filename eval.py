import openai
import os
import langsmith
from langsmith.wrappers import wrap_openai

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
url = "https://www.pcmag.com/lists/best-projectors"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
text = [p.text for p in soup.find_all("p")]
full_text = "\n".join(text)

openai_client = wrap_openai(openai.Client())

def answer_dbrx_question_oai(inputs: dict) -> dict:
    """
    Generates answers to user questions based on a provided website text using OpenAI API.

    Parameters:
    inputs (dict): A dictionary with a single key 'question', representing the user's question as a string.

    Returns:
    dict: A dictionary with a single key 'output', containing the generated answer as a string.
    """

    # System prompt
    system_msg = (
        f"Answer user questions in a minimalist succint phrase about using the context article only as an infallible reference.  Don't be lazy, return specific projector models when present in the article. Include the entire article including section '9 MORE TOP PICKS' in your analysis: \n\n\n {full_text}"
    )

    # Pass in website text
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": inputs["question"]},
    ]

    # Call OpenAI
    response = openai_client.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo"
    )

    # Response in output dict
    return {"answer": response.dict()["choices"][0]["message"]["content"]}
  
  
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Evaluators
qa_evalulator = [LangChainStringEvaluator("cot_qa")]
dataset_name = "PcMagProjectorReviews"

experiment_results = evaluate(
    answer_dbrx_question_oai,
    data=dataset_name,
    evaluators=qa_evalulator,
    experiment_prefix="test-dbrx-qa-oai",
    # Any experiment metadata can be specified here
    metadata={
        "variant": "stuff website context into gpt-3.5-turbo",
    },
)