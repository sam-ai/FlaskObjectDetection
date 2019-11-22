import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
import re, time, base64
from io import BytesIO
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
MODEL_NAME = 'frozen_bol_ADDRESS'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'cws_label.pbtxt')
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def getI420FromBase64(codec):
    # data:image/png;base64
    """ Convert image from a base64 bytes stream to an image. """
    base64_data = re.sub(b'^data:image/.+;base64,', b'', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img

def load_image_into_numpy_array(image):

  (im_width, im_height) = image.size
  # if image.getdata().mode == "P":
  #     image = image.convert('RGB')
  image = image.convert('RGB')
  np_array = np.array(image.getdata())
  reshaped = np_array.reshape((im_height, im_width, 3))

  return reshaped.astype(np.uint8), (im_width, im_height)


def crop_img_to_bytes(image_np, 
                      boxes, 
                      scores,
                      img_class,
                      im_width,
                      im_height,
                      min_score_thresh=.5):
  
  class_mappining = {
    1 : 'ADDRESS'
  }
  
  crop_images = {}
  
  for i in range(min(20, boxes.shape[0])):
    if scores is None or scores[0][i] > min_score_thresh:
      box = tuple(boxes[0][i].tolist())
      i_class = class_mappining[img_class[0][i]] + "_" + str(i)
      ymin, xmin, ymax, xmax = box
      (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
      a,b,c,d = int(left) , int(right) , int(top) ,int(bottom)
      arr = image_np[c:d,a:b]
      img = Image.fromarray(arr)
      imgByteArr = BytesIO()
      img.save(imgByteArr, format='PNG')
      imgByteArr = imgByteArr.getvalue()
      crop_images[i_class] = base64.b64encode(imgByteArr).decode('utf-8')

  return crop_images



app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


        
@app.route('/detection', methods=['POST'])
def detection():
    request.get_data()
    
    # Load in an image to object detect and preprocess it
    img_data = getI420FromBase64(request.data)
    image_np, shape = load_image_into_numpy_array(img_data)
    print(shape)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # x_input = np.expand_dims(img_data, axis=0) # add an extra dimention.

    result = {
        'data': "Done !"
    }

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
                      [boxes, scores, classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
          # print("scores -> {}".format(scores))
          # print("classes -> {}".format(classes))
          # print("boxes -> {}".format(boxes))
    
          # vis_util.visualize_boxes_and_labels_on_image_array(
          #             image_np,
          #             np.squeeze(boxes),
          #             np.squeeze(classes).astype(np.int32),
          #             np.squeeze(scores),
          #             category_index,
          #             use_normalized_coordinates=True,
          #             line_thickness=3)
          # crop_img_to_bytes(image_np, output_dict['detection_boxes'],
          #         output_dict['detection_scores'], output_dict['detection_classes'], 1692, 2200)
          data = crop_img_to_bytes(image_np,
                                   boxes,
                                   scores,
                                   classes,
                                   shape[0],
                                   shape[1])
    # im = Image.fromarray(image_np)

    return jsonify(data)

    


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
#     TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
#     IMAGE_SIZE = (12, 8)

#     with detection_graph.as_default():
#         with tf.Session(graph=detection_graph) as sess:
#             for image_path in TEST_IMAGE_PATHS:
#                 image = Image.open(image_path)
#                 image_np = load_image_into_numpy_array(image)
#                 image_np_expanded = np.expand_dims(image_np, axis=0)
#                 image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#                 boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#                 scores = detection_graph.get_tensor_by_name('detection_scores:0')
#                 classes = detection_graph.get_tensor_by_name('detection_classes:0')
#                 num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#                 (boxes, scores, classes, num_detections) = sess.run(
#                     [boxes, scores, classes, num_detections],
#                     feed_dict={image_tensor: image_np_expanded})
#                 vis_util.visualize_boxes_and_labels_on_image_array(
#                     image_np,
#                     np.squeeze(boxes),
#                     np.squeeze(classes).astype(np.int32),
#                     np.squeeze(scores),
#                     category_index,
#                     use_normalized_coordinates=True,
#                     line_thickness=8)
#                 im = Image.fromarray(image_np)
#                 im.save('uploads/'+filename)

#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
