from gradio_client import Client
import os

def check_robys_api():
    print("Checking Robys01/Face-Aging API structure...")
    try:
        client = Client("Robys01/Face-Aging")
        print("\n--- DETAILED API VIEW ---")
        client.view_api()
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    check_robys_api()
