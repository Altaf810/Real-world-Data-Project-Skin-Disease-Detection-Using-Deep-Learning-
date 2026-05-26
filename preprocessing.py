import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


data_dir = r"D:\Skin diease detection"
img_dir1 = os.path.join(data_dir, "HAM10000_images_part_1")
img_dir2 = os.path.join(data_dir, "HAM10000_images_part_2")


metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
df = pd.read_csv(metadata_path)

print("✅ Unique disease labels:", df['dx'].unique())


label_mapping = {label: idx for idx, label in enumerate(df['dx'].unique())}
print("✅ Label Mapping:", label_mapping)


IMG_SIZE = 128
images, labels, missing = [], [], []


for i, row in df.iterrows():
    img_id = row['image_id']
    
    
    possible_files = [
        os.path.join(img_dir1, img_id + ".jpg"),
        os.path.join(img_dir1, img_id + ".png"),
        os.path.join(img_dir2, img_id + ".jpg"),
        os.path.join(img_dir2, img_id + ".png"),
    ]
    
    img_path = None
    for path in possible_files:
        if os.path.exists(path):
            img_path = path
            break
    
    if img_path is None:
        missing.append(img_id)
        continue  
    
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  
    
    images.append(img)
    labels.append(label_mapping[row['dx']])

print(f"✅ Total images loaded: {len(images)}")
print(f"⚠️ Missing images: {len(missing)}")


X = np.array(images, dtype=np.float32)
y = np.array(labels)


y = to_categorical(y, num_classes=len(label_mapping))

print("✅ Dataset shape:", X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Training set:", X_train.shape, y_train.shape)
print("✅ Testing set:", X_test.shape, y_test.shape)


save_path = os.path.join(data_dir, "preprocessed_128.npz")
np.savez_compressed(
    save_path,
    X_train=X_train.astype(np.float16),
    X_test=X_test.astype(np.float16),
    y_train=y_train.astype(np.uint8),
    y_test=y_test.astype(np.uint8)
)

print(f"✅ Data preprocessing done and saved to: {save_path}")
