import cv2
import numpy as np
import os
from ultralytics import YOLO
from sklearn.cluster import KMeans


class RoseTesterFinal:
    def __init__(self, model_path, output_dir="final_test_results"):
        print(f"正在加载模型: {model_path}")
        self.detector = YOLO(model_path)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_radial_ratio(self, keypoints):
        """1. 几何特征提取 - 计算 Rgeo"""
        kp0 = keypoints[0, :2]
        inner = keypoints[1:5, :2]
        outer = keypoints[5:9, :2]
        d_in = np.mean(np.linalg.norm(inner - kp0, axis=1))
        d_out = np.mean(np.linalg.norm(outer - kp0, axis=1))
        r_geo = (d_out - d_in) / (d_out + 1e-6)
        return r_geo, d_out

    def analyze_roi_cpa(self, img, center_kp, d_out):
        """2. ROI-CPA 光度分析[cite: 1, 3]"""
        h, w = img.shape[:2]
        cx, cy = int(center_kp[0]), int(center_kp[1])
        r_roi = 0.5 * d_out  # λ=0.5

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r_roi), 255, -1)
        pixels = cl[mask > 0].reshape(-1, 1)
        if pixels.size < 10: return 0.0, 1.0, mask

        kmeans = KMeans(n_clusters=3, n_init=3, random_state=42).fit(pixels)
        shadow_val = np.min(kmeans.cluster_centers_)
        shadow_mask_raw = (cl < (shadow_val + 20)).astype(np.uint8)
        final_shadow_mask = cv2.bitwise_and(shadow_mask_raw, mask)

        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        sigma = 0.8 * r_roi
        weight_map = np.exp(-dist_sq / (2 * (sigma ** 2)))

        r_photo = np.sum(weight_map[final_shadow_mask > 0]) / (np.sum(weight_map[mask > 0]) + 1e-6)

        mask_light = cv2.bitwise_and(mask, cv2.bitwise_not(final_shadow_mask * 255))
        i_light = np.mean(cl[mask_light > 0]) if np.sum(mask_light) > 0 else 255
        i_dark = np.mean(cl[final_shadow_mask > 0]) if np.sum(final_shadow_mask) > 0 else 1
        contrast = i_light / (i_dark + 1e-5)

        return r_photo, contrast, final_shadow_mask * 255

    def draw_flower_visual(self, img, box, kpts, final_grade, r_geo, r_photo=None, contrast=None, shadow_mask=None,
                           d_out=0):
        """3. 全要素可视化"""
        x1, y1, x2, y2 = map(int, box)
        kp0_coord = (int(kpts[0, 0]), int(kpts[0, 1]))
        grade_map = {1: "Level1(Bud)", 2: "Level2(Start)", 3: "Level3(Semi)", 4: "Level4(Flat)", 5: "Level5(Open)"}

        lw = 10
        pt_radius = 12
        t_dynamic = 0.8 * r_geo - 0.35

        if shadow_mask is not None:
            overlay = img.copy()
            overlay[shadow_mask > 0] = (255, 255, 0)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            cv2.circle(img, kp0_coord, int(0.5 * d_out), (0, 0, 255), 5)

        for p in kpts[1:5, :2]:
            cv2.line(img, kp0_coord, (int(p[0]), int(p[1])), (255, 0, 0), lw)
            cv2.circle(img, (int(p[0]), int(p[1])), pt_radius, (255, 0, 0), -1)
        for p in kpts[5:9, :2]:
            cv2.line(img, kp0_coord, (int(p[0]), int(p[1])), (0, 255, 0), lw)
            cv2.circle(img, (int(p[0]), int(p[1])), pt_radius, (0, 255, 0), -1)
        cv2.circle(img, kp0_coord, pt_radius, (0, 0, 255), -1)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 12)

        top_str = f"{grade_map[final_grade]} (Rg:{r_geo:.3f})"
        cv2.putText(img, top_str, (x1, y1 - 190), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 255, 255), 8)

        if r_photo is not None:
            data_str = f"Tr:{t_dynamic:.2f} Rp:{r_photo:.2f} C:{contrast:.1f}"
            cv2.putText(img, data_str, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 8)

    def process_image(self, img_path):
        """核心判别流程 - 重新整理的级联逻辑[cite: 3]"""
        img = cv2.imread(img_path)
        if img is None: return

        results = self.detector(img, verbose=False)[0]
        if not results.boxes: return

        for i in range(len(results.boxes)):
            box = results.boxes.xyxy[i].cpu().numpy()
            kpts = results.keypoints.data[i].cpu().numpy()

            # A. 宏观几何初筛
            r_geo, d_out = self.calculate_radial_ratio(kpts)
            r_photo, contrast, shadow_mask = None, None, None

            if r_geo >= 0.8:
                final_grade = 1
            elif 0.7 < r_geo < 0.8:
                final_grade = 2
            elif 0.55 <= r_geo <= 0.7:
                # B. 微观光度精判 (针对三级与四级的核心逻辑更新)
                r_photo, contrast, shadow_mask = self.analyze_roi_cpa(img, kpts[0, :2], d_out)
                t_dynamic = 0.8 * r_geo - 0.35

                # 1. 强特征锁定：高对比度直接确认为三级[cite: 3]
                if contrast > 2.8:
                    final_grade = 3
                # 2. 典型特征区：中等对比度带判定为四级[cite: 3]
                elif 2.0 <= contrast <= 2.8:
                    final_grade = 4
                # 3. 形态补偿区：低对比度情况下由 Rp 决定 (含 C < 2.0 的所有情况)[cite: 3]
                else:
                    if r_photo > t_dynamic:
                        final_grade = 3  # 形态精准锁定为受干扰的三级
                    else:
                        final_grade = 4  # 分散阴影判定为四级
            else:
                final_grade = 5

            self.draw_flower_visual(img, box, kpts, final_grade, r_geo, r_photo, contrast, shadow_mask, d_out)

        save_path = os.path.join(self.output_dir, f"result_{os.path.basename(img_path)}")
        cv2.imwrite(save_path, img)
        print(f"分析完成: {save_path}")


if __name__ == "__main__":
    MODEL = r'best.pt'
    IMAGE = r'2.jpg'

    tester = RoseTesterFinal(MODEL)
    tester.process_image(IMAGE)