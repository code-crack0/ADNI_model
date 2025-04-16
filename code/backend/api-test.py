import requests
import os
import time
import json
import base64
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import traceback

# Configuration
NIFTI_FILE_PATH = "./mri-images/grouped-images-test-nii/AD/ADNI_002_S_0938_MR_MPR-R__GradWarp__B1_Correction_Br_20070713122900520_S29621_I60044.nii"
API_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = "./test_results"
SAVE_RESULTS = True
DEBUG_MODE = True  # Set to True for more detailed error information

def test_api():
    """Test all functionality of the NIfTI classifier API with detailed error reporting"""
    print("=" * 60)
    print("TESTING NIFTI CLASSIFIER API WITH GRADCAM")
    print("=" * 60)
    
    if SAVE_RESULTS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Step 1: Test API health
    if not test_api_health():
        return
    
    # Step 2: Test prediction with error handling
    try:
        test_prediction_with_error_handling()
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        if DEBUG_MODE:
            traceback.print_exc()

def test_api_health():
    """Test if the API is running and accessible"""
    print("\n" + "-" * 60)
    print("STEP 1: Testing API health")
    print("-" * 60)
    
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ API is running")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Failed to connect to API at {API_URL}")
        print("Make sure the API is running and the URL is correct")
        return False

def test_prediction_with_error_handling():
    """Test prediction with detailed error handling to identify failing steps"""
    print("\n" + "-" * 60)
    print("STEP 2: Testing prediction with detailed error handling")
    print("-" * 60)
    
    if not os.path.exists(NIFTI_FILE_PATH):
        print(f"❌ File not found: {NIFTI_FILE_PATH}")
        return
    
    print(f"📁 Using NIfTI file: {os.path.basename(NIFTI_FILE_PATH)}")
    
    # Create a session to maintain headers and settings
    session = requests.Session()
    
    # Set a longer timeout to avoid timeouts during processing
    timeout = 300  # 5 minutes
    
    # Prepare file for upload
    with open(NIFTI_FILE_PATH, "rb") as file:
        files = {"file": (os.path.basename(NIFTI_FILE_PATH), file, "application/octet-stream")}
        
        print("⏳ Uploading and processing file...")
        start_time = time.time()
        
        try:
            # Use debug endpoint mode if available
            url = f"{API_URL}/predict/?debug=true" if DEBUG_MODE else f"{API_URL}/predict/"
            response = session.post(url, files=files, timeout=timeout)
            
            elapsed_time = time.time() - start_time
            print(f"⏱️  Request completed in {elapsed_time:.2f} seconds")
            
            # Save raw response for debugging
            if SAVE_RESULTS and DEBUG_MODE:
                with open(os.path.join(OUTPUT_DIR, "raw_response.txt"), "w") as f:
                    f.write(f"Status code: {response.status_code}\n\n")
                    f.write(f"Headers: {response.headers}\n\n")
                    f.write(f"Content: {response.text}\n")
                print(f"💾 Raw response saved to: {os.path.join(OUTPUT_DIR, 'raw_response.txt')}")
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                print_prediction_results(result)
                save_results(result)
                display_heatmap(result)
                return result
            else:
                print(f"\n❌ Prediction failed with status code {response.status_code}")
                print(f"Error: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"\n❌ Request timed out after {timeout} seconds")
            print("The processing might be taking longer than expected.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Request error: {e}")
            return None
        except json.JSONDecodeError:
            print("\n❌ Failed to parse response as JSON")
            print(f"Raw response: {response.text[:500]}...")  # Show first 500 chars
            return None

def print_prediction_results(result):
    # """Display prediction results in a formatted way"""
    # print("\n✅ PREDICTION SUCCESSFUL")
    # print("-" * 30)
    # print(f"🏷️  Predicted class: {result['predicted_class']}")
    # print(f"🔢 Confidence: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
    
    # if 'heatmap' in result:
    #     print("🔍 GradCAM heatmap received")
    # else:
    #     print("❌ No heatmap returned in the response")

    """Display prediction results in a formatted way"""
    print("\n✅ PREDICTION SUCCESSFUL")
    print("-" * 30)
    print(f"🏷️  Predicted class: {result['predicted_class']}")
    print(f"🔢 Confidence: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
    
    if 'heatmap' in result:
        print("🔍 GradCAM heatmap received with region identification")
        print("   - Heatmap shows the top 3 regions influencing the prediction")
        print("   - Red/yellow areas indicate stronger influence on classification")
    else:
        print("❌ No heatmap returned in the response")

def save_results(result):
    """Save prediction results to disk"""
    if not SAVE_RESULTS:
        return
        
    base_filename = os.path.basename(NIFTI_FILE_PATH).replace('.nii.gz', '').replace('.nii', '')
    
    # Save prediction result as JSON
    result_file = os.path.join(OUTPUT_DIR, f"{base_filename}_result.json")
    with open(result_file, 'w') as f:
        # Remove the base64 string from the JSON to keep it small
        result_copy = result.copy()
        if 'heatmap' in result_copy:
            result_copy['heatmap'] = "[base64 data removed]"
        json.dump(result_copy, f, indent=2)
    print(f"💾 Prediction result saved to: {result_file}")
    
    # Save heatmap image if present
    if 'heatmap' in result:
        try:
            image_data = base64.b64decode(result['heatmap'])
            heatmap_image = Image.open(BytesIO(image_data))
            
            heatmap_file = os.path.join(OUTPUT_DIR, f"{base_filename}_heatmap.png")
            heatmap_image.save(heatmap_file)
            print(f"💾 Heatmap image saved to: {heatmap_file}")
        except Exception as e:
            print(f"❌ Failed to save heatmap: {e}")

def display_heatmap(result):
    """Display the heatmap using matplotlib"""
    # if 'heatmap' not in result:
    #     return
        
    # try:
    #     image_data = base64.b64decode(result['heatmap'])
    #     heatmap_image = Image.open(BytesIO(image_data))
        
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(heatmap_image)
    #     plt.title(f"GradCAM Heatmap - Predicted: {result['predicted_class']} ({result['probability']*100:.2f}%)")
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    # except Exception as e:
    #     print(f"❌ Failed to display heatmap: {e}")

    """Display the enhanced GradCAM heatmap with labeled regions"""
    if 'heatmap' not in result:
        return
        
    try:
        image_data = base64.b64decode(result['heatmap'])
        heatmap_image = Image.open(BytesIO(image_data))
        
        plt.figure(figsize=(12, 10))  # Slightly larger figure to accommodate labels
        plt.imshow(heatmap_image)
        plt.title(f"Enhanced GradCAM Analysis - Predicted: {result['predicted_class']} ({result['probability']*100:.2f}%)\n"
                 f"Highlighted regions show areas most important for classification", 
                 fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        
        # Add explainer text below plot
        plt.figtext(0.5, 0.01, 
                   "Numbered circles mark the top 3 regions influencing the model's decision",
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.7, "pad":5})
        
        plt.show()
    except Exception as e:
        print(f"❌ Failed to display heatmap: {e}")

if __name__ == "__main__":
    test_api()