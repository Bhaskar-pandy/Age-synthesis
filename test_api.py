from gradio_client import Client
import os

def test_api():
    print("Connecting to D-X-Lab/Face-Aging...")
    try:
        client = Client("D-X-Lab/Face-Aging")
        print("API Endpoints:")
        client.view_api()
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_api()
