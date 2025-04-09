import requests
import json
import base64
import time
from PIL import Image
import io
import os


def connect_to_battery_detection_server(ngrok_url):
    """
    Connect to the battery detection server and test the health endpoint

    Args:
        ngrok_url (str): The ngrok URL of the server (without the /detect endpoint)

    Returns:
        bool: True if connected successfully, False otherwise
    """
    try:
        # Make sure the URL is properly formatted
        if not ngrok_url.startswith("http"):
            ngrok_url = "https://" + ngrok_url

        # Remove trailing slash if present
        if ngrok_url.endswith("/"):
            ngrok_url = ngrok_url[:-1]

        # Test the health endpoint
        health_url = f"{ngrok_url}/health"
        response = requests.get(health_url, timeout=10)

        if response.status_code == 200:
            health_data = response.json()
            print(f"Successfully connected to server! Status: {health_data['status']}")
            print(f"Model loaded: {health_data['model_loaded']}")
            return True
        else:
            print(f"Server returned error status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return False


def request_battery_detection(ngrok_url):
    """
    Send a request to the battery detection server to capture and process an image

    Args:
        ngrok_url (str): The ngrok URL of the server (without the /detect endpoint)

    Returns:
        dict: The detection results or None if failed
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

        print("\nSending detection request to server...")
        start_time = time.time()

        # Send POST request to the detection endpoint
        response = requests.post(detect_url, timeout=30)

        elapsed_time = time.time() - start_time
        print(f"Received response in {elapsed_time:.2f} seconds")

        if response.status_code == 200:
            result_data = response.json()

            # Check if the request was successful
            if result_data.get('success', False):
                print("Detection completed successfully!")

                # Save the processed image
                if 'processed_image' in result_data:
                    image_data = base64.b64decode(result_data['processed_image'])
                    image = Image.open(io.BytesIO(image_data))

                    # Create results directory if it doesn't exist
                    os.makedirs('results', exist_ok=True)

                    # Save image with timestamp
                    timestamp = int(time.time())
                    image_path = f"results/processed_image_{timestamp}.jpg"
                    image.save(image_path)
                    print(f"Saved processed image to {image_path}")

                    # Display the image if running in an environment that supports it
                    try:
                        image.show()
                    except:
                        print("Note: Unable to display image automatically in this environment")

                # Process the detection results
                print("\n=== Detection Results ===")
                print(f"Number of detections: {len(result_data['detections'])}")

                if len(result_data['detections']) > 0:
                    print("\nBattery detections:")
                    for i, det in enumerate(result_data['detections']):
                        print(f"  {i + 1}. {det['class_name']} (confidence: {det['confidence']:.2f})")
                        print(f"     Position: x={det['displacement']['x']:.1f}, y={det['displacement']['y']:.1f}")
                        print(
                            f"     Offsets: small_x={det['small_x_offset']:.2f}, small_y={det['small_y_offset']:.2f}, small_angle={det['small_angle']:.2f}")

                print("\nClosest object:")
                print(json.dumps(result_data['closest_object'], indent=2))

                print("\nImage dimensions:")
                print(
                    f"Width: {result_data['image_dimensions']['width']}, Height: {result_data['image_dimensions']['height']}")

                return result_data
            else:
                print(f"Server reported an error: {result_data.get('error', 'Unknown error')}")
                return None
        else:
            print(f"Server returned error status code: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"Error during detection request: {e}")
        return None


def main():
    print("Battery Detection System Client")
    print("==============================")

    # Get ngrok URL from user
    ngrok_url = input("Enter the ngrok URL from your Colab notebook: ")

    # Connect to server
    if connect_to_battery_detection_server(ngrok_url):
        while True:
            # Ask user if they want to send a detection request
            user_input = input("\nSend detection request? (yes/no): ").lower()

            if user_input in ["yes", "y"]:
                # Send detection request
                detection_results = request_battery_detection(ngrok_url)

                # Ask if user wants to save the results to a JSON file
                if detection_results:
                    save_option = input("\nSave results to JSON file? (yes/no): ").lower()
                    if save_option in ["yes", "y"]:
                        # Create results directory if it doesn't exist
                        os.makedirs('results', exist_ok=True)

                        # Save results with timestamp
                        timestamp = int(time.time())
                        json_path = f"results/detection_results_{timestamp}.json"

                        # Remove the base64 image from the saved results to reduce file size
                        results_to_save = detection_results.copy()
                        if 'processed_image' in results_to_save:
                            del results_to_save['processed_image']

                        with open(json_path, 'w') as f:
                            json.dump(results_to_save, f, indent=2)

                        print(f"Results saved to {json_path}")

            elif user_input in ["no", "n"]:
                print("Exiting...")
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    print("Battery Detection Client terminated.")


if __name__ == "__main__":
    main()