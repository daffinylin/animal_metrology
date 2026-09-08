import math
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Path
BASE_DIR = Path(r"D:\Wiwynn\segmentation_animal")
IMAGE_DIR = BASE_DIR / "images"
MODEL_DIR = BASE_DIR / "models"

SEG_MODEL_PATH = MODEL_DIR / "yolov8x-seg.pt"
POSE_MODEL_PATH = MODEL_DIR / "ap10k_pt.pth"
YOLO_POSE_MODEL_PATH = MODEL_DIR / "yolov8x-pose.pt"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ANIMAL_CLASS_IDS = [ 15, 16, 17, 18, 19, 20, 21, 22, 23 ]  #  (貓, 狗, 馬, 羊, 牛, 象, 熊, 斑馬, 長頸鹿)


def load_ap10k_pose_model(pth_path: Path):

    base_model = YOLO(YOLO_POSE_MODEL_PATH)

    if pth_path.exists():
        try:
            checkpoint = torch.load(
                str(pth_path), map_location="cpu", weights_only=False
            )

            if "model" in checkpoint:
                model_obj = checkpoint["model"]
                state_dict = (
                    model_obj.state_dict()
                    if hasattr(model_obj, "state_dict")
                    else model_obj
                )
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            base_model.model.load_state_dict(state_dict, strict=False)
            print("Load ap10k_pt.pth！")
        except Exception as e:
            print(
                f"Load ap10k_pt.pth error ({e}),apply yolov8x-pose"
            )
    else:
        print(f"Cant find {pth_path}, apply yolov8x-pose")

    return base_model


def calculate_distance(p1, p2):
    """Euclidean Distance"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def rotate_image_and_get_matrix(image, angle):
    """image rotate 0, 90, 180, 270"""
    h, w = image.shape[:2]
    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        M_inv = np.array([[0, 1, 0], [-1, 0, w]], dtype=np.float32)
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        M_inv = np.array([[-1, 0, w], [0, -1, h]], dtype=np.float32)
    elif angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        M_inv = np.array([[0, -1, h], [1, 0, 0]], dtype=np.float32)
    else:
        rotated = image.copy()
        M_inv = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    return rotated, M_inv


def transform_kpt_back_matrix(x, y, M_inv):
    """back to the original ROI image"""
    pt = np.array([x, y, 1.0], dtype=np.float32)
    orig_pt = np.dot(M_inv, pt)
    return float(orig_pt[0]), float(orig_pt[1])


# def enhance_eye_features(crop_img):
#     """CLAHE, highlighting the details and shine"""
#     lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl, a, b))
#     return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def enhance_eye_features(crop_img):
    """Turn to grayscake and CLAHE, highlighting the details and shine"""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return enhanced_bgr


def process_image(img, seg_model, pose_model):
    annotated_img = img.copy()
    h_orig, w_orig = img.shape[:2]

    seg_results = seg_model(img, verbose=False)[0]
    detected_animals = []

    if seg_results.boxes is not None and seg_results.masks is not None:
        for box, mask in zip(seg_results.boxes, seg_results.masks.data):
            cls_id = int(box.cls[0])
            if cls_id not in ANIMAL_CLASS_IDS:
                continue

            # object contours
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (w_orig, h_orig))
            contours, _ = cv2.findContours(
                mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated_img, contours, -1, (0, 255, 0), 2)

            # increase 35% Padding instead of cut off
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            bw, bh = x2 - x1, y2 - y1

            # close to the left
            pad_left = int(bw * 0.4) if x1 < w_orig * 0.1 else int(bw * 0.25)
            pad_top = int(bh * 0.25)
            pad_right = int(bw * 0.25)
            pad_bottom = int(bh * 0.25)

            rx1, ry1 = max(0, x1 - pad_left), max(0, y1 - pad_top)
            rx2, ry2 = min(w_orig, x2 + pad_right), min(h_orig, y2 + pad_bottom)

            roi_img = img[ry1:ry2, rx1:rx2]
            if roi_img.size == 0:
                continue

            # feature enhance
            roi_enhanced = enhance_eye_features(roi_img)

            #  TTA +  Conf 
            angles = [0, 90, 270, 180]
            best_eyes = {}
            max_conf_score = -1.0

            for angle in angles:
                rot_roi, M_inv = rotate_image_and_get_matrix(
                    roi_enhanced, angle
                )

                # imgsz and conf
                pose_results = pose_model(
                    rot_roi, imgsz=1280, conf=0.0001, verbose=False
                )[0]

                if (
                    pose_results.keypoints is None
                    or len(pose_results.keypoints) == 0
                ):
                    continue

                kpts_all = pose_results.keypoints.xy.cpu().numpy()
                confs_all = pose_results.keypoints.conf.cpu().numpy()

                for kpts, confs in zip(kpts_all, confs_all):
                    l_idx, r_idx = (1, 2) if len(kpts) >= 17 else (0, 1)

                    l_conf = confs[l_idx] if len(confs) > l_idx else 0.0
                    r_conf = confs[r_idx] if len(confs) > r_idx else 0.0
                    total_score = l_conf + r_conf

                    if total_score > max_conf_score:
                        max_conf_score = total_score
                        current_eyes = {}

                        if l_conf > 0.005:
                            lx, ly = transform_kpt_back_matrix(
                                kpts[l_idx][0], kpts[l_idx][1], M_inv
                            )
                            current_eyes["left_eye"] = (
                                int(rx1 + lx),
                                int(ry1 + ly),
                            )

                        if r_conf > 0.005:
                            rx, ry = transform_kpt_back_matrix(
                                kpts[r_idx][0], kpts[r_idx][1], M_inv
                            )
                            current_eyes["right_eye"] = (
                                int(rx1 + rx),
                                int(ry1 + ry),
                            )

                        best_eyes = current_eyes

            if best_eyes:
                detected_animals.append(best_eyes)

    # draw eyes 
    for idx, animal in enumerate(detected_animals):
        if "left_eye" in animal and "right_eye" in animal:
            l_eye, r_eye = animal["left_eye"], animal["right_eye"]
            dist = calculate_distance(l_eye, r_eye)

            cv2.line(annotated_img, l_eye, r_eye, (0, 255, 255), 2)
            cv2.circle(annotated_img, l_eye, 5, (0, 0, 255), -1)
            cv2.circle(annotated_img, r_eye, 5, (0, 0, 255), -1)

            mid_x, mid_y = (l_eye[0] + r_eye[0]) // 2, (l_eye[1] + r_eye[1]) // 2
            cv2.putText(
                annotated_img,
                f"A{idx+1}: {dist:.1f}px",
                (mid_x - 35, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

    # right_eye to right_eye(object from left to right)
    right_eyes = [a["right_eye"] for a in detected_animals if "right_eye" in a]
    right_eyes = sorted(right_eyes, key=lambda p: p[0])  

    if len(right_eyes) >= 2:
        r1, r2 = right_eyes[0], right_eyes[1]
        inter_dist = calculate_distance(r1, r2)
        cv2.line(annotated_img, r1, r2, (255, 0, 255), 2, cv2.LINE_AA)
        mid_x, mid_y = (r1[0] + r2[0]) // 2, (r1[1] + r2[1]) // 2
        cv2.putText(
            annotated_img,
            f"Inter R-Eye: {inter_dist:.1f}px",
            (mid_x - 50, mid_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    return annotated_img



def main():
    if not SEG_MODEL_PATH.exists():
        print(f"Can't find {SEG_MODEL_PATH} yolov8x-seg.pt")
        return

    print("Initial Model...")
    seg_model = YOLO(str(SEG_MODEL_PATH))
    pose_model = load_ap10k_pose_model(POSE_MODEL_PATH)

    image_paths = [
        p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in VALID_EXTS
    ]
    if not image_paths:
        print(f"在 {IMAGE_DIR} Empty")
        return

    OUTPUT_DIR = Path(r"D:\Wiwynn\segmentation_animal\output")

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue

        result_img = process_image(img, seg_model, pose_model)

        output_filename = f"{img_path.stem}_output{img_path.suffix}"
        output_path = OUTPUT_DIR / output_filename

       
        cv2.imwrite(str(output_path), result_img)

        # Show the result
    #     cv2.imshow("Animal Metrology Analysis", result_img)
    #     key = cv2.waitKey(0)
    #     if key == ord("q"):
    #         break

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()