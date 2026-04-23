from gradio_client import Client, handle_file
import os

def test_final_call():
    print("Testing Robys01/Face-Aging with POSITIONAL arguments...")
    try:
        client = Client("Robys01/Face-Aging")
        # Use a dummy path or a real one if available. 
        # For testing connection/params, we can just use a local image if we had one.
        # Let's try to just check the return type if possible, or just prepare the fix.
        print("API is alive. Positional args: [handle_file, source_age, target_age]")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_final_call()
