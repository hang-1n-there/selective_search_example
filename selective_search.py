import selective_search
import cv2
import matplotlib.pyplot as plt
import os
import selectivesearch
import numpy as np

class Img():
    def img_read(self, img):
        
        self.img = cv2.imread(img)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        print('img.shape : ', self.img.shape)

        plt.figure(figsize=(8,8))
        plt.imshow(self.img_rgb)
        plt.show()
        
        return self.img_rgb
    
class Region():
    def __init__(self, img_rgb):
        self.img_rgb = img_rgb
        
    def region_proposal(self):
        # Region proposal을 추출
        _, self.regions = selectivesearch.selective_search(self.img_rgb, scale=100, min_size=2000)
        print(type(self.regions), len(self.regions))
        
    def rect(self):
        # rect 정보만 추출
        self.cand_rects = [cand['rect'] for cand in self.regions]
        print(self.cand_rects)
        
        return self.cand_rects
    
    def bbox(self):
        green_rgb = (125, 255, 51)
        self.img_rgb_copy = self.img_rgb.copy()
        
        for rect in self.cand_rects:
            left, top = rect[0], rect[1]
            
            #rect[2] : 너비 , rect[3] : 높이
            right = left + rect[2]
            bottom = top + rect[3]
            
            self.img_rgb_copy = cv2.rectangle(self.img_rgb_copy, (left,top) , (right,bottom) , color = green_rgb , thickness= 2)
         
        plt.figure(figsize=(8,8))
        plt.imshow(self.img_rgb_copy)
        plt.show()               
        return self.img_rgb_copy

class IoU():
    def __init__(self, cand_box, gt_box):
        self.gt_box = gt_box # ground truth bounding box
        self.cand_box = cand_box
        
    def compute_iou(self):
        # intersection : x_max , y_max , x_min , y_min
        
        x1 = np.maximum(self.cand_box[0], self.gt_box[0])
        y1 = np.maximum(self.cand_box[1], self.gt_box[1])
        x2 = np.minimum(self.cand_box[2], self.gt_box[2])
        y2 = np.minimum(self.cand_box[3], self.gt_box[3])
        
        cand_box_area = (self.cand_box[2] - self.cand_box[0]) * (self.cand_box[3] - self.cand_box[1])
        gt_box_area = (self.gt_box[2] - self.gt_box[0]) * (self.gt_box[3] - self.gt_box[1])
        
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = cand_box_area + gt_box_area - intersection
        
        iou = intersection / union
        
        return iou

class Gt_box():
    def __init__(self, img_rgb):
        self.img_rgb = img_rgb
        
    def GT_box(self):
        red = (255,0,0)
        gt_box = [190, 21, 390, 211] #fine tuned
        
        self.img_rgb = cv2.rectangle(self.img_rgb , (gt_box[0], gt_box[1]) , (gt_box[2], gt_box[3]), color=red, thickness= 2)
        plt.figure(figsize=(8,8))
        plt.imshow(self.img_rgb)
        plt.show()
        
        return gt_box

class Visualization_():
    def __init__(self, cand_rects, gt_box, img_rgb):
        self.cand_rects = cand_rects
        self.gt_box = gt_box
        self.img_rgb = img_rgb
        
    def visualization(self):
        green = (125, 255, 51)
        
        # 생성자 선언
        iou_calculator = IoU(None, None)
        for index , cand_box in enumerate(cand_rects):
            cand_box = list(cand_box)
            
            cand_box[2] += cand_box[0]
            cand_box[3] += cand_box[1]
            
            # IoU 클래스의 인스턴스를 생성하고 compute_iou() 메서드 호출
            iou_calculator.cand_box = cand_box
            iou_calculator.gt_box = self.gt_box
            iou = iou_calculator.compute_iou()
            
            if iou > 0.5:
                print('index :', index, 'iou :', iou, 'rectangle :', (cand_box[0],cand_box[1],cand_box[2],cand_box[3]))
                cv2.rectangle(self.img_rgb, (cand_box[0],cand_box[1]) , (cand_box[2], cand_box[3]) , color=green , thickness=1)
                text = "{} : {:.2f}".format(index, iou)
                cv2.putText(self.img_rgb, text, (cand_box[0]+100, cand_box[1]+10) , cv2.FONT_HERSHEY_SIMPLEX, 0.4, color = green, thickness=1)
        
        plt.figure(figsize=(12,12))
        plt.imshow(self.img_rgb)
        plt.show()
        
        return self.img_rgb
if __name__ == "__main__":
    img = '0. Img/IU.jpg'
    img_obj = Img()  # Img 클래스의 인스턴스 생성
    img_rgb = img_obj.img_read(img)  # img_read() 메서드 호출
    
    region = Region(img_rgb)
    region.region_proposal()
    cand_rects = region.rect() # rect만 추출
    img_rgb_copy = region.bbox() # bbox 시각화
    
    box = Gt_box(img_rgb)
    gt_box = box.GT_box() # gt_box
    
    Visual = Visualization_(cand_rects,gt_box,img_rgb)
    Visual_img = Visual.visualization()