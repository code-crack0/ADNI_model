import requests

url = "http://localhost:5000/predict/"
image_path = r".\MCI - ADNI_029_S_0878_MR_MPR-R__GradWarp__B1_Correction_Br_20100111155051603_S76982_I163064.png"

# Open image file in binary mode
with open(image_path, "rb") as img_file:
    files = {"file": ("image.png", img_file, "image/png")}  # Explicitly set content type
    response = requests.post(url, files=files)

# Print response
print(response.json())

