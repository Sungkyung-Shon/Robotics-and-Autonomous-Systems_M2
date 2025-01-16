from rasrobot import RASRobot
import numpy as np
import time
import torch
from pathlib import Path
# from yolov5.models.experimental import attempt_load
# from yolov5.utils.datasets import letterbox
# from yolov5.utils.general import non_max_suppression, scale_coords
import cv2
import yolov5
#A quick and effective real-time object recognition system built on deep learning is called YOLO (You Only Look Once). 
#It does a single pass processing of the full image, gridding it to determine bounding boxes and class probabilities. 
#The accuracy and speed of YOLO have improved with each new version that has been launched.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
class MyRobot(RASRobot):
    #Define the __init__() method, which loads the YOLOv5 model and initialises the class attributes.
    def __init__(self):
        super(MyRobot, self).__init__()
        # Initialise and resize a new window 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolov5_model = yolov5.load('yolov5s.pt') # Load YOLOv5s model
        self.yolov5_model.conf = 0.3  # Confidence threshold
        self.yolov5_model.iou = 0.45  # NMS IoU threshold
        self.yolov5_model.classes = [0]  # Only detect class 0 (stop sign)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 128*4, 64*4)
        self.stopIdentifiedBool=0
        self.stopFlag=0
    #The robot's camera is used to take a picture while creating a mask to help it identify the road's yellow lines.
    def get_road(self):
        image = self.get_camera_image()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 100, 100], np.uint8)
        upper = np.array([40, 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    #yolov5 model can find stop signs in the images captured by cameras.
    #After that, the robot stops in few seconds when it sees a stop sign.
    # def detect_stop_sign(self, frame):
        # img = letterbox(frame, self.yolov5_model.img_size)[0]
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)
        # img = torch.from_numpy(img).to(self.device).float() / 255.0  # uint8 to float32, 0-255 to 0-1
        # img = img.unsqueeze(0)  # Add batch dimension
        
        # results = model(img)
        # predictions = results.pred[0]
        # boxes = predictions[:, :4] # x1, y1, x2, y2
        # scores = predictions[:, 4]
        # categories = predictions[:, 5]

        # with torch.no_grad():
            # pred = self.yolov5_model(img)[0]  # Pass the image through the model
            # pred = non_max_suppression(pred, self.yolov5_model.conf, self.yolov5_model.iou, self.yolov5_model.classes)

        # stop_sign_detected = False

        # for det in pred:  # Detections per image
            # if det is not None and len(det):
                # stop_sign_detected = True
                # break

        # return stop_sign_detected
        
    #Use the yolo model for detecting stop signs in the camera image.
    #When a stop sign is detected, the robot stops for few seconds.
    def detect_stop_sign(self):
       self.loop=13
       image=self.get_camera_image()           
       # Check if the image data is None
       if image is None: 
           print("Not Detected")
       else:    
           results = model(image)
           stop_sign_coords = None
           for result in results.xyxy[0]:
               if result[5] == 11:  # 11 is the class id for stop sign in YOLOv5
                    stop_sign_coords = result[:4]
                    break
           if stop_sign_coords is not None:               
               if self.stopIdentifiedBool==0:
                   if self.stopFlag==0:
                       for _ in range(self.loop):
                           self.loop -= 1
                       if self.loop==0:
                           print("StopDetected")
                           time.sleep(2)
                           self.stopFlag=1
                           self.stopIdentifiedBool=1
                           
                    #self.stopFlag=1
                    #img = cv2.rectangle(img, (int(stop_sign_coords[0]), int(stop_sign_coords[1])), (int(stop_sign_coords[2]), int(stop_sign_coords[3])), (0, 255, 0), 2)
                    #cv2.imshow("image", img) 
               else:
                   print("-") 
                   self.stopIdentifiedBool=0
                  
                   #self.stopFlag=0
                   #self.stopFlag=0
    #Analyzes the masked image of the yellow line and calculates the steering angle to follow the line.
    def yellowline(self, edges):
        
        #Follow the yellow line by turning left or right depending on where the line is in the image.
        
        # Crop the image to only look at the bottom quarter
        # restricting the view since zebra cross is in yello
        height, width = edges.shape
        cropped = edges[3*height//4:height,:]
        # Get the indices of the white pixels
        indices = np.where(cropped == 255)
        
        # Check if there are any white pixels in the image
        if len(indices[0]) == 0:
            return 0
        # Compute the center of the white pixels
        center = np.mean(indices[1])
        #print(center)
        # Compute the deviation from the center of the image
        deviation = center - width/2
        #print(deviation)
        # Compute the steering angle
        steering_angle = deviation/(width/2)
        return steering_angle 

    #It is the robot's main loop. The robot's speed and steering angle are controlled in accordance with the camera image after it has been captured, processed to identify yellow lines, and stop signs.
    def run(self):
        
        #This function implements the main loop of the robot.
        
        while self.tick():
            # Get the camera image and convert it to grayscale
            image = self.get_camera_image()            
            image = cv2.resize(image, (600,600))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert to LAB color space
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # create a CLAHE object
            image[:,:,0] = clahe.apply(image[:,:,0])  # apply CLAHE to the L channel
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)  # convert back to BGR color space
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            yellowLL = np.array([20, 60, 100])
            #upper_yellow = np.array([35, 167, 166])
            #yellowUL = np.array([36, 160, 162])
            yellowUL = np.array([30, 130, 200])
            mask_yellow = cv2.inRange(hsv, yellowLL, yellowUL)
            #print(mask_yellow)
            # Apply Gaussian blur to mask
            mask_blur = cv2.GaussianBlur(mask_yellow, (5, 5), 0)
            #print(mask_blur)
            # Apply Canny edge detection to mask
            edges = cv2.Canny(mask_blur, 100, 200)
            
            steering_angle = self.yellowline(edges)
            
   
            self.detect_stop_sign()
                        
            # If the yellow line ends, turn right
            if steering_angle == 0:
                steering_angle = 0.4
                speed = 30
                self.stopFlag=0
                #default_angle=steering_angle
             # thresholding the steering angle to avoid drift
            if steering_angle < -0.3:
                steering_angle =- 0.3
                speed = 40 
                #default_angle=steering_angle
            elif steering_angle > 0.3:
                steering_angle = 0.3
                speed = 40 
                #default_angle=steering_angle
            else:
                 steering_angle=0
                 speed = 40 
                
                       
            # Set the speed and steering angle of the robot
            self.set_speed(speed)
            self.set_steering_angle(steering_angle)
            # Display the output image with the detected edges
            output = np.dstack((edges, edges, edges))
            cv2.imshow('output', output)
            cv2.waitKey(1)


# We just create an instance and let it do its job.
robot = MyRobot()
robot.run()


