from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, DlibWrapper, ArcFace
# from accelerate import Accelerator, init_empty_weights
import pdb
import json
import shutil
import pandas as pd
import os
import argparse
import random
import pickle
from pathlib import Path

random.seed(3141)

from tqdm import tqdm

# 대충 초당 7~8장 처리 가능

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser(description='Face Recognition Model')
parser.add_argument('--model', type=str, default='Facenet512', help='Model Name')
parser.add_argument('--backend', type=str, default='retinaface', help='Detector Backend')
parser.add_argument('--result-dir', type=str, default='demo/results', help='Image Result Directory')
# parser.add_argument('--semi-result-dir', type=str, default='examples/results/semi_result', help='Semi Image Result Directory')
# parser.add_argument('--result-file', type=str, default='examples/results/result.json', help='Result File')
# parser.add_argument('--semi-result-file', type=str, default='examples/results/semi_result.json', help='Semi Result File')
parser.add_argument('--target-image', type=str, default='examples/inputs/001_fe3347c0.jpg', help='Target Image')
parser.add_argument('--db-path', type=str, default='examples/dbs/cfd_all', help='Database Path')
parser.add_argument('--reference-classes', type=str, default='examples/dbs/cfd_all_names.txt', help='Reference Classes')
parser.add_argument('--set-threshold', action='store_true', default=False, help='set threshold')
parser.add_argument('--distance', type=str, default='cosine', help='Distnace Metric')


args = parser.parse_args()

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



with open(args.reference_classes, 'r') as f: 
  datas = f.readlines()
new_data = {}
for data in datas:
  data = data.strip().replace('\n', '').split()
  new_data[data[0]] = int(data[1])


analyze_file_name = f"analyze_{args.model}.pkl"
if os.path.isfile(os.path.join(args.db_path, analyze_file_name)):
  analyze_file = pickle.load(open(os.path.join(args.db_path, analyze_file_name), 'rb'))
else:
  analyze_file = {}
  for file in tqdm(os.listdir(args.db_path)):
      if ((".jpg" in file.lower()) or (".jpeg" in file.lower()) or (".png" in file.lower())):
        objs = DeepFace.analyze(img_path = os.path.join(args.db_path, file),
            actions = ['gender', 'race',],
            detector_backend=args.backend,
            enforce_detection=False,)
        analyze_file[file] = objs
    
  with open(os.path.join(args.db_path, analyze_file_name), 'wb') as f:
    pickle.dump(analyze_file, f)

# pdb.set_trace()

if os.path.isfile(args.target_image):
  target_files = [args.target_image]
elif os.path.isdir(args.target_image):
  target_files = [os.path.join(args.target_image, x) for x in os.listdir(args.target_image)]
else:
  raise Exception("target_image is not file or directory")

for target_file in tqdm(target_files):
  target_result_dir = os.path.join(args.result_dir, Path(target_file).stem)
  os.makedirs(target_result_dir, exist_ok=True)
  
  target_class = new_data[os.path.basename(target_file)]
  # pdb.set_trace()

  objs = DeepFace.analyze(img_path = target_file, 
          # actions = ['age', 'gender', 'race', 'emotion']
          actions = ['gender', 'race',],
          enforce_detection=False,
          )

  dfs= DeepFace.find(img_path = target_file, 
                      db_path = args.db_path,
                      model_name = args.model,
                      enforce_detection=False,
                      detector_backend=args.backend,
                      set_threshold=args.set_threshold,
                      distance_metric=args.distance,
                      )

  new_dfs = []
  for df in dfs:
    df['category'] = df['identity'].map(lambda x : new_data[x])
    df = df[df["category"] != target_class]
    df = df.drop_duplicates(subset=['category'], keep='first')
    df = df.reset_index(drop=True)
    new_dfs.append(df.to_dict('index'))
  
  semi_result_file = os.path.join(target_result_dir, 'semi_result.json')
  with open(semi_result_file, 'w') as f:
    json.dump(new_dfs, f, indent=4)
    
  for i, df in enumerate(new_dfs):
    os.makedirs(os.path.join(target_result_dir, 'semi_result', str(i)), exist_ok=True)
    for idx in df:
      shutil.copy(os.path.join(args.db_path, df[idx]['identity']), os.path.join(target_result_dir, 'semi_result', str(i), f"{idx}_{df[idx]['category']}_{df[idx]['identity']}"))

  final_result = []
  for i, df in tqdm(enumerate(new_dfs)):
    if len(objs) <= i:
      break
    os.makedirs(os.path.join(target_result_dir, 'result', str(i)), exist_ok=True)
    
    for k_idx in df:
        k_objs = analyze_file[df[k_idx]['identity']]
        
        for k_obj in k_objs:
          # if (k_obj['dominant_gender'] == objs[i]['dominant_gender']):
          if (k_obj['dominant_race'] == objs[i]['dominant_race']) and  (k_obj['dominant_gender'] == objs[i]['dominant_gender']):
            overall_aspect = {}
            overall_aspect.update(df[k_idx])
            overall_aspect.update(k_obj)
            shutil.copy(os.path.join(args.db_path, df[k_idx]['identity']), os.path.join(target_result_dir, 'result', str(i), f"{k_idx}_{df[k_idx]['category']}_{df[k_idx]['identity']}"))
            final_result.append(overall_aspect)
  
  result_file = os.path.join(target_result_dir, 'result.json')
  with open(result_file, 'w') as f:
    json.dump(final_result, f, indent=4)
    
  # pdb.set_trace()