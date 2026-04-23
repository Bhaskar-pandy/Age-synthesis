from gradio_client import Client
import sys

def capture_api():
    with open("api_schema.txt", "w") as f:
        sys.stdout = f
        client = Client("Robys01/Face-Aging")
        client.view_api()
    sys.stdout = sys.__stdout__
    print("API schema captured to api_schema.txt")

if __name__ == "__main__":
    capture_api()
