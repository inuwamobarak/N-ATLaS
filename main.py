from model_utils import NAtlasModel
import os
from dotenv import load_dotenv

load_dotenv()
def main():
    model = NAtlasModel()

    # Example Hausa query. refer to notebook for more queries
    q_chat = [
        {
            "role": "system",
            "content": "You are a large language model trained by Awarri AI technologies. You are a friendly assistant and you are here to help."
        },
        {
            "role": "user",
            "content": "menene ake nufi da gwagwarmaya"  # Hausa: "What is meant by struggle?"
        }
    ]

    response = model.chat(q_chat)
    print("\n=== Model Response ===\n")
    print(response)


if __name__ == "__main__":
    main()