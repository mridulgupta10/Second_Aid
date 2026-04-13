import requests
import io
from PIL import Image

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

url = "http://127.0.0.1:3000/detect"
files = {'file': ('dummy.jpg', img_byte_arr, 'image/jpeg')}

response = requests.post(url, files=files)
print("Status Code:", response.status_code)
print("Response:", response.text)
