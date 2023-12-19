from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, DlibWrapper, ArcFace
from accelerate import Accelerator, init_empty_weights
import pdb
import json
import shutil
import pandas as pd
import os
from tqdm import tqdm

# 대충 초당 7~8장 처리 가능

import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]


target_image = "/home/monchana/new_project/face_prj/data/celeba/001_fe3347c0.jpg"
db_path = "/home/monchana/new_project/face_prj/cfd_all"
reference_classes = '/home/monchana/new_project/face_prj/cfd_names.txt'


semi_result_dir = "semi_result"
result_dir = "result"
semi_result_file = "semi_result.json"
result_file = "result.json"
if os.path.isdir(result_dir):
  shutil.rmtree(result_dir)
if os.path.isdir(semi_result_dir):
  shutil.rmtree(semi_result_dir)

# accelerator = Accelerator()
# model = VGGFace.loadModel()
# model = accelerator.prepare(model)
# target_image = "/home/monchana/new_project/face_prj/data/celeba/001198.jpg"
# db_path = "/home/monchana/new_project/face_prj/samples_test"
# reference_classes = '/home/monchana/new_project/face_prj/data/celeba/identity_CelebA.txt'


with open(reference_classes, 'r') as f: 
  datas = f.readlines()
new_data = {}
for data in datas:
  data = data.strip().replace('\n', '').split()
  new_data[data[0]] = int(data[1])

target_class = new_data[os.path.basename(target_image)]
# new_data = []
# for data in datas:
#   data = data.strip().replace('\n', '').split()
#   new_data.append([data[0], int(data[1])])

objs = DeepFace.analyze(img_path = target_image, 
        actions = ['age', 'gender', 'race', 'emotion']
)

dfs= DeepFace.find(img_path = target_image, 
                    db_path = db_path,
                    model_name = models[2],
                    enforce_detection=False,
                    detector_backend='retinaface',
                    set_threshold=False
                    )

new_dfs = []
for df in dfs:
  df['category'] = df['identity'].map(lambda x : new_data[x])
  df = df[df["category"] != target_class]
  df = df.drop_duplicates(subset=['category'], keep='first')
  df = df.reset_index(drop=True)
  new_dfs.append(df.to_dict('index'))
  
with open(semi_result_file, 'w') as f:
  json.dump(new_dfs, f)
  
for i, df in enumerate(new_dfs):
  os.makedirs(os.path.join(semi_result_dir, str(i)), exist_ok=True)
  os.makedirs(os.path.join(result_dir, str(i)), exist_ok=True)
  for idx in df:
    shutil.copy(os.path.join(db_path, df[idx]['identity']), os.path.join(semi_result_dir, str(i), f"{idx}_{df[idx]['category']}_{df[idx]['identity']}"))



final_result = []
for i, df in tqdm(enumerate(new_dfs)):
  for k_idx in df:
      k_objs = DeepFace.analyze(img_path = os.path.join(db_path, df[k_idx]['identity']), 
                                actions = ['gender', 'race'], enforce_detection=False)
      for k_obj in k_objs:
        if (k_obj['dominant_race'] == objs[i]['dominant_race']) and  (k_obj['dominant_gender'] == objs[i]['dominant_gender']):
          shutil.copy(os.path.join(db_path, df[k_idx]['identity']), os.path.join(result_dir, str(i), f"{k_idx}_{df[k_idx]['category']}_{df[k_idx]['identity']}"))
          final_result.append(df[k_idx])
        
with open(result_file, 'w') as f:
  json.dump(final_result, f)
  
  
# # dfs = dfs.values.tolist()
# with open('result.json', 'w') as f:
#   json.dump(his, f)

pdb.set_trace()