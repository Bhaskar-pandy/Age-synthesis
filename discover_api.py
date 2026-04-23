from gradio_client import Client
import time

SPACES = [
    "Robys01/Face-Aging",
    "D-X-Lab/Face-Aging",
    "nhndev/FRAN",
    "Ziqin/Face-Aging"
]

def discover_api():
    for space in SPACES:
        print(f"Checking {space}...")
        try:
            # Try to connect with a short timeout to see if it's public
            client = Client(space)
            print(f"SUCCESS: {space} is PUBLIC.")
            print("Endpoints:")
            client.view_api()
            return space
        except Exception as e:
            print(f"FAILED: {space} - {e}")
            print("-" * 20)
    return None

if __name__ == "__main__":
    discover_api()
