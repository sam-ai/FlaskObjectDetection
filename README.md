# Flask Object Detector - TensorFlow

[![](images/logo.png)](https://www.tensorflow.org/)

## Using:

```
- git clone
- save frozen model(other files also) in FlaskObjectDetection/frozen_bol_ADDRESS folder
- cd FlaskObjectDetection
- pip install -r requirements.txt
- python app.y

```


## API USAGE:

```python
url = 'https://<api url host>/detection'
img_path = "/<path to image>/R1178.png"
  
import base64
import requests

encoded = b'data:image/png;base64,' + base64.b64encode(open(img_path, "rb").read())
res_out = requests.post(url, data = encoded)
crop_img = res_out.json()
crop_img['data']
```

## Response:

```
{
  "data": [
    {
      "base64png": "iVCAzzfe//+tcVj959OGKinllZaXd3d3DwyO2TRcvrl......",
      "bounding_box": [
        128,
        858,
        147,
        352
      ],
      "label": "ADDRESS"
    },
    {
      "base64png": "iVBORw0KGgoAAAANSUhEUgAAAtcAAADQCAIAAABDdtqbAAEAAEl......",
      "bounding_box": [
        130,
        857,
        360,
        568
      ],
      "label": "ADDRESS"
    }
  ],
  "title": "address_predictor"
}
```

## API CLIENT:

get cropped images

```

def get_cropped_address(img_path):
  base_url = 'https://5bc01aed.ngrok.io/detection'
  encoded = b'data:image/png;base64,' + \
            base64.b64encode(open(img_path, "rb").read())
  res_out = requests.post(base_url, data = encoded)
  crop_img = res_out.json()['data']
  
  return [
          Image.open(BytesIO(base64.b64decode(img['base64png']))) 
          for img in crop_img
         ]
         
```

get ocr text

```

def get_text_ocr(data):
  import pytesseract
  for im in data:
    im = im.convert('L')
    text = pytesseract.image_to_string(im)
    print(text)
    print("=="*10)

```

test

```

data = get_cropped_address(img_path)
get_text_ocr(data)

```


