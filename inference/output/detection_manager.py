import pandas as pd
import datetime
#YOLOV5s
CLASS_PER_FRAME={
    0:{
        "name":"person",
        "frame_detections":0
    },
    1:{
        "name":"bicycle",
        "frame_detections":0
    },
    2:{
        "name":"car",
        "frame_detections":0
    },
    3:{
        "name":"motorbike",
        "frame_detections":0
    },
    5:{
        "name":"bus",
        "frame_detections":0
    },
    7:{
        "name":"truck",
        "frame_detections":0
    }
}

STRUCT_DATA =   {
    'obj_id': None, 
    'obj_class': None, 
    'score': None, 
    'first_bbox': None, 
    'last_bbox': None,
    'input_zone': None,
    'output_zone':None,
    'frames_counter':None,
    'frames_counter_per_clss':None,    
    'detection_time':None
    }
TEMP_FINISH_TIMER_MINUTES = 1
TEMP_FINISH_TIMER_SECONDS = 10
class DetectionManager:
    def __init__(self):

        self.zones = []
        self.detections = []
        self.detections_dataframe = None
        self.count_timer = datetime.datetime.now()

    def obj_exists(self,obj_id,obj_class):
        for detection in self.detections:
            if detection['obj_id'] == obj_id:
                detection['frames_counter']+=1
                detection['frames_counter_per_clss'][obj_class]['frame_detections']+=1
                return True                
        return False

    def print_obj(self,obj_id):
        for detection in self.detections:
            print(detection['obj_id']== obj_id)            
        return False

    def update(self,bbox,obj_id,obj_class):
        aux_detection =  STRUCT_DATA.copy()
        
        if len(self.detections)>0 and self.count_timer + datetime.timedelta( seconds = TEMP_FINISH_TIMER_SECONDS) < datetime.datetime.now():
            self.count_timer = datetime.datetime.now()
            self.detections_dataframe = pd.DataFrame(self.detections)
            self.detections_dataframe.to_csv("Data_"+self.count_timer.strftime("%d%m%y_%H_%M_%S")+".csv",sep=':')
        
        if self.obj_exists(obj_id,obj_class):
            return


        aux_detection['obj_id']  = obj_id
        aux_detection['obj_class']  = obj_class
        aux_detection['first_bbox']  = str(bbox)
        aux_detection['last_bbox']  = str(bbox)
        aux_detection['frames_counter']  = 0
        aux_detection['input_zone']  = -1
        aux_detection['output_zone']  = -1
        aux_detection['frames_counter_per_clss']  = CLASS_PER_FRAME
        aux_detection['detection_time']  = datetime.datetime.now()
        self.detections.append(aux_detection)
        print(len(self.detections))

