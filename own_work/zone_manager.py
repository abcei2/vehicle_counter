import cv2
import copy
import numpy as np
import json

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class DesignatedArea:
    def __init__(self, inputZone):
        self.zone = Polygon( [inputZone[0], inputZone[1], inputZone[2], inputZone[3]] )
        self.points = [inputZone[0], inputZone[1], inputZone[2], inputZone[3]]
        self.count = 0
        self.allowed = True

    def contains(self, inPoint):
        return self.zone.contains( Point( inPoint)  )

class ZoneConfig:
    def __init__(self,ref_frame=[],load_config=True,path_config="./own_work/zone_config.json"):
        self.ref_frame = ref_frame.copy()
        self.draw_frame = ref_frame.copy()
        self.mouse_x = -1
        self.mouse_y = -1
        self.pressed = False
        self.configured = False
        self.load_config = load_config
        self.path_config = path_config
        self.north_poly = {
            "p1":{
                "point":[50 ,50],
                "pressed":False
            },            
            "p2":{
                "point":[50 ,100],
                "pressed":False
            },            
            "p3":{
                "point":[100 ,100],
                "pressed":False
            },            
            "p4":{
                "point":[100 ,50],
                "pressed":False
            }
        }
        
        self.south_poly =copy.deepcopy(self.north_poly)
        self.west_poly =copy.deepcopy(self.north_poly)
        self.east_poly =copy.deepcopy(self.north_poly)
        self.center_poly =copy.deepcopy(self.north_poly)
    
        if load_config:
            with open(path_config) as json_file:
                data = json.load(json_file)
                if "north_poly" in data:
                    self.north_poly = data["north_poly"]
                if "south_poly" in data:
                    self.south_poly = data["south_poly"]
                if "west_poly" in data:
                    self.west_poly = data["west_poly"]
                if "east_poly" in data:
                    self.east_poly = data["east_poly"]   
                if "center_poly" in data:
                    self.center_poly = data["center_poly"]      
        self.areas = None
        self.update_designated_area()
        self.draw_status ={
            0:"S",
            1:"N",
            2:"W",
            3:"E",
            4:"C",
        }
        self.draw_counter = 0
        
        self.configured = False

    def point_inside_area(self, point):
        to_return = {        
            "north_poly":False,
            "south_poly":False,
            "west_poly":False,
            "east_poly":False,            
        }
        for key_poly in self.areas:   
            if self.areas[key_poly].contains(point):         
                to_return[key_poly] = True
                break
        return to_return

        
    def update_designated_area(self):
        self.areas = {
            "north_poly":self.get_designated_area(self.north_poly), 
            "south_poly":self.get_designated_area(self.south_poly), 
            "west_poly":self.get_designated_area(self.west_poly), 
            "east_poly":self.get_designated_area(self.east_poly), 
            "center_poly":self.get_designated_area(self.center_poly),
        }
    def get_designated_area(self, poly):
        return DesignatedArea([ poly[key_point]["point"] for key_point in poly])



    def check_click_distance(self,p1,p2):
        distance = ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
        if distance < 30:
            return True
        else:
            return False
            
    def draw(self, event,x,y,flags,param):
        if self.configured:
            return
        self.draw_frame = self.ref_frame.copy()
        self.mouse_x = x
        self.mouse_y = y
        status = self.draw_status[self.draw_counter]
        actual_poly = self.north_poly

        if status=="N":
            actual_poly = self.north_poly
        elif status=="S":
            actual_poly = self.south_poly
        elif status=="E":
            actual_poly = self.east_poly
        elif status=="W":
            actual_poly = self.west_poly
        elif status=="C":
            actual_poly = self.center_poly


        if event == cv2.EVENT_RBUTTONDOWN:
            self.draw_counter += 1
            if self.draw_counter > 4:
                self.draw_counter = 0

        if event == cv2.EVENT_MOUSEMOVE:
            if self.pressed:                
                for k_points in actual_poly:    
                    if actual_poly[k_points]["pressed"]:          
                        actual_poly[k_points]["point"]=[x,y]                              

        if event == cv2.EVENT_LBUTTONUP:      
            self.pressed = False            
            for k_points in actual_poly:            
                actual_poly[k_points]["pressed"]=False  

        if event == cv2.EVENT_LBUTTONDOWN:
            self.pressed = True      
            for k_points in actual_poly:    
                if self.check_click_distance([x,y],actual_poly[k_points]["point"]):     
                    actual_poly[k_points]["point"]=[x,y]                    
                    actual_poly[k_points]["pressed"]=True 
                    break        
                           
        self.draw_frame = self.draw_poly(self.draw_frame,actual_poly)

    def draw_zones(self,img):

        img = self.draw_poly(img,self.north_poly)
        img = self.draw_poly(img,self.south_poly)
        img = self.draw_poly(img,self.west_poly)
        img = self.draw_poly(img,self.east_poly)
        img = self.draw_poly(img,self.center_poly)
        return img

    def draw_poly(self,img,actual_poly):
        # Initialize black image of same dimensions for drawing the rectangles
        blk = np.zeros(img.shape, np.uint8)
        # Draw rectangles
        cv2.fillPoly(blk,self.to_polyline(actual_poly),(0,255,255))     
        img = cv2.addWeighted(img, 1.0, blk, 0.25, 1)
        cv2.polylines( img,self.to_polyline(actual_poly),True,(0,255,255),2)
        return img
      
    def to_polyline(self, actual_poly):
        pts = np.array([ actual_poly[k_points]["point"] for k_points in actual_poly], np.int32)
        pts = pts.reshape((-1,1,2))
        return [pts]

    def update_image(self, img):
        self.ref_frame = img.copy()
        self.draw_frame = self.ref_frame.copy()


    def configure_system_coordinates(self):
        
        if self.configured:
            return
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw)
        print("LET'S CALIBRATE")
        while cv2.waitKey(10)!=ord('q'):
            cv2.imshow('image',self.draw_frame)
  
        self.configured = True
        self.update_designated_area()
        cv2.destroyAllWindows()
        if self.load_config:
            with open(self.path_config, 'w') as outfile:
                data = {
                    "north_poly": self.north_poly,
                    "south_poly": self.south_poly,
                    "west_poly": self.west_poly,
                    "east_poly": self.east_poly,
                    "center_poly": self.center_poly,
                }
                json.dump(data, outfile)

        print("END CALIBRATE")