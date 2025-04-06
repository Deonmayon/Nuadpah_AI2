from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv
import uuid
import numpy as np
import cv2
import os
import requests

app = FastAPI()

load_dotenv()

model = YOLO("model/new_best_seg.pt")

output_dir = "output_images/"
os.makedirs(output_dir, exist_ok=True)

ref_image_path = "images/person_15.jpg"
ref_image = cv2.imread(ref_image_path)
ref_h, ref_w, _ = ref_image.shape

label_file = "labels/person_15_shoulder.txt"
ref_keypoints = []

with open(label_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        x_center, y_center, width, height = parts[1:5]
        keypoints = []
        
        for i in range(5, len(parts), 3):  # Each keypoint has x, y, conf
            if i + 2 < len(parts):  # Ensure we have all 3 values
                kx, ky, conf = parts[i], parts[i+1], parts[i+2]
                keypoints.append((kx, ky, conf))
        
        # Calculate bounding box coordinates
        ref_x_min = (x_center - width/2) * ref_w
        ref_y_min = (y_center - height/2) * ref_h
        ref_bbox_width = width * ref_w
        ref_bbox_height = height * ref_h
        
        # Convert keypoints to relative coordinates
        keypoint_list = []
        for kx, ky, conf in keypoints:
            rel_x = (kx * ref_w - ref_x_min) / ref_bbox_width
            rel_y = (ky * ref_h - ref_y_min) / ref_bbox_height
            rel_x = max(0, min(1, rel_x))
            rel_y = max(0, min(1, rel_y))
            keypoint_list.append((rel_x, rel_y, conf))
        
        ref_keypoints.append((class_id, keypoint_list))

def get_sequential_keypoint_pairs(keypoints_list):
    all_line_pairs = []
    for class_id, keypoints in keypoints_list:
        line_pairs = []
        num_points = len(keypoints)
        if num_points > 1:
            mid_point = num_points // 2
            for i in range(0, mid_point - 1, 2):  # Step in groups of 2
                line_pairs.append((i, i + 1))
                if i + mid_point < num_points - 1:
                    line_pairs.append((i + mid_point, i + 1 + mid_point))
        all_line_pairs.append(line_pairs)   
    return all_line_pairs
        
def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def draw_keypoints_on_target(img_np, results):
    frame_h, frame_w, _ = img_np.shape
    output_frame = img_np.copy()
    sequential_pairs = get_sequential_keypoint_pairs(ref_keypoints)

    TARGET_CLASS_NAME = "shoulder"
    base_radius = int(12 * (frame_w / 800))
    base_radius = max(5, min(base_radius, 30))

    GREEN = (0, 255, 0)
    GREEN_DARK = (0, 200, 0)

    target_masks = []

    for r in results:
        if r.masks is None:
            continue
        class_indices = r.boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[idx] for idx in class_indices]

        for i, (mask, class_name, conf) in enumerate(zip(r.masks.xy, class_names, r.boxes.conf)):
            if class_name == TARGET_CLASS_NAME:
                target_masks.append((mask, conf.item(), r.boxes.xyxy[i].cpu().numpy()))

    if not target_masks:
        return output_frame, "No target class detected."

    # Use the most confident detection
    target_masks.sort(key=lambda x: x[1], reverse=True)
    mask, confidence, box = target_masks[0]
    mask_np = np.array(mask, dtype=np.int32)

    x_min, y_min = np.min(mask_np, axis=0)
    x_max, y_max = np.max(mask_np, axis=0)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    for idx, (class_id, keypoints) in enumerate(ref_keypoints):
        # Draw sequential keypoint connections
        for pair in sequential_pairs[idx]:
            k1, k2 = pair
            kx1, ky1, _ = keypoints[k1]
            kx2, ky2, _ = keypoints[k2]

            mapped_x1 = int(x_min + kx1 * bbox_width)
            mapped_y1 = int(y_min + ky1 * bbox_height)
            mapped_x2 = int(x_min + kx2 * bbox_width)
            mapped_y2 = int(y_min + ky2 * bbox_height)

            if cv2.pointPolygonTest(mask_np, (mapped_x1, mapped_y1), False) >= 0 and \
               cv2.pointPolygonTest(mask_np, (mapped_x2, mapped_y2), False) >= 0:
                cv2.line(output_frame, (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), GREEN_DARK , 2)
                
        for kx_rel, ky_rel, conf in keypoints:
            mapped_x = int(x_min + kx_rel * bbox_width)
            mapped_y = int(y_min + ky_rel * bbox_height)
            if cv2.pointPolygonTest(mask_np, (mapped_x, mapped_y), False) >= 0:
                cv2.circle(output_frame, (mapped_x, mapped_y), base_radius, GREEN, -1)

    return output_frame, None

@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.post("/predict")
# async def predict(url: str):
#     try:
#         # Get image from URL
#         img = get_image(url)

#         # Convert to numpy (YOLO expects ndarray or path)
#         img_np = np.array(img)

#         # Predict using YOLO
#         results = model(img_np)

#         # Extract result data
#         result_data = []
#         for result in results:
#             boxes = result.boxes
#             masks = result.masks
#             for i in range(len(boxes)):
#                 box = boxes[i].xyxy.cpu().numpy().tolist()[0] if boxes is not None else None
#                 conf = boxes[i].conf.item() if boxes is not None else None
#                 cls = int(boxes[i].cls.item()) if boxes is not None else None
#                 class_name = model.names[cls] if cls is not None else None

#                 result_data.append({
#                     "class_id": cls,
#                     "class_name": class_name,
#                     "confidence": conf,
#                     "bbox": box,
#                 })

#         return {
#             "success": True,
#             "predictions": result_data
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e)
#         }
        
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Fix color

        results = model(img_bgr)
        output_img, error = draw_keypoints_on_target(img_bgr, results)

        if error:
            return {"success": False, "message": error}

        file_name = f"pred_{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(output_dir, file_name)
        cv2.imwrite(save_path, output_img)
        
        # create supabase client
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_API_KEY")
        )
        
        with open(save_path, "rb") as f:
            file_data = f.read()
            storage_path = f"predictions/{datetime.utcnow().isoformat()}_{file_name}"
            supabase.storage.from_(os.getenv("SUPABASE_BUCKET_NAME")).upload(
                storage_path,
                file_data,
                {"content-type": "image/jpeg"}
            )
            public_url = supabase.storage.from_(os.getenv("SUPABASE_BUCKET_NAME")).get_public_url(storage_path)

        return {
            "success": True,
            "message": "Keypoints drawn and image saved.",
            "output_image": save_path,
            "public_url": public_url["publicURL"]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}