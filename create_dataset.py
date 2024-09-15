from langsmith import Client
from dotenv import load_dotenv

import os

client = Client()
print( f"Client: {client}")
dataset_name = "PcMagProjectorReviewsXXX"


inputs = [
    "what is the highest ranked home entertainment projector?",
    "what is the highest rating for any projector?",
    "how many top picks are there?",
    "what is the best mini projector?",
    "what is the best ultra short throw projector?",
    "what is the best projector for a home theater?",
    "what is the best 1080p business projector?",
    "what is the best gaming projector?",
    "what is the best rugged indoor/outdoor projector?",
    "what is the best budget home entertainment projector?",
  #  'what is the google page rank of the context article link, normalized from 0-1?'
    "how legitimate is the article, normalized from 0-1?"
]

outputs = [
    "BenQ TK860i",
    "5.0",
    "10",
    "Anker Nebula Capsule 3 Laser",
    "Epson EpiqVision Ultra LS800 3-Chip 3LCD Smart Streaming Laser Projector",
    "Epson Pro Cinema LS12000 4K Pro-UHD Laser Projector",
    "BenQ LH730",
    "BenQ X3100i",
    "Anker Nebula Mars 3",
    "Vankyo Performance V700W",
    "0.5"
]

# Store
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="QA pairs about pcmag.com projector article.",
)
client.create_examples(
    inputs=[{"question": q} for q in inputs],
    outputs=[{"answer": a} for a in outputs],
    dataset_id=dataset.id,
)
