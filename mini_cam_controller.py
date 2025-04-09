import requests
import json
import time


def get_camera_offsets(ngrok_url, timeout=10):
    """
    Connect to the battery detection server and get the offsets

    Args:
        ngrok_url (str): The ngrok URL of the server (without the /detect endpoint)
        timeout (int): Timeout in seconds for the request

    Returns:
        dict: The offsets dictionary with small_x, small_y, small_angle or None if failed
    """
    try:
        # Make sure the URL is properly formatted
        if not ngrok_url.startswith("http"):
            ngrok_url = "https://" + ngrok_url

        # Remove trailing slash if present
        if ngrok_url.endswith("/"):
            ngrok_url = ngrok_url[:-1]

        # Build the detection endpoint URL
        detect_url = f"{ngrok_url}/detect"

        print("\n[Camera Client] Sending detection request to server...")
        start_time = time.time()

        # Send POST request to the detection endpoint
        response = requests.post(detect_url, timeout=timeout)

        elapsed_time = time.time() - start_time
        print(f"[Camera Client] Received response in {elapsed_time:.2f} seconds")

        if response.status_code == 200:
            result_data = response.json()

            # Check if the request was successful
            if result_data.get('success', False) and 'closest_object' in result_data:
                print("[Camera Client] Detection completed successfully!")

                # Extract the offset values from the closest object
                closest_object = result_data['closest_object']

                # Create an offsets dictionary with the values we need
                offsets = {
                    'small_x': closest_object.get('small_x_offset', 0.0),
                    'small_y': closest_object.get('small_y_offset', 0.0),
                    'small_angle': closest_object.get('small_angle', 0.0),
                    'class_name': closest_object.get('class_name', 'unknown'),
                    'confidence': closest_object.get('confidence', 0.0)
                }

                print(
                    f"[Camera Client] Offsets: small_x={offsets['small_x']:.2f}, small_y={offsets['small_y']:.2f}, small_angle={offsets['small_angle']:.2f}")
                print(f"[Camera Client] Class: {offsets['class_name']} (confidence: {offsets['confidence']:.2f})")

                return offsets
            else:
                print(f"[Camera Client] Server reported an error: {result_data.get('error', 'Unknown error')}")
                return None
        else:
            print(f"[Camera Client] Server returned error status code: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"[Camera Client] Error during detection request: {e}")
        return None


if __name__ == "__main__":
    # Test the function
    ngrok_url = input("Enter the ngrok URL from your Colab notebook: ")
    offsets = get_camera_offsets(ngrok_url)
    if offsets:
        print("Test successful!")
        print(json.dumps(offsets, indent=2))
    else:
        print("Test failed!")