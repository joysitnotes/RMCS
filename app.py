import cv2
import os 
import time
import datetime



#Camera setup
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 100
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 100
cap.set(cv2.CAP_PROP_FRAME_WIDTH,530) #640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,350) #480
#cap.set(cv2.CAP_PROP_FPS, 15)

frame_size = (int(cap.get(3)), int(cap.get(4)))

#Output video type
#fourcc = cv2.VideoWriter_fourcc(*"avc1")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#Save Directory - path to hardrive
outDir = '/PATH/TO/EXTERNAL/DRIVE' +'/CCTV/'
#Set Frame rate
frameRate = 20

#Main Function loop
def CCTV():
    loop = True
    clipNumber = 0

    while loop:
        frame_time = int(0)
        #File and folder name variable and checks
        current_day = datetime.datetime.now().strftime("%d-%m-%Y")
        today_directory = outDir + current_day
        
        if not os.path.exists(today_directory):
            os.makedirs(today_directory)   

                

        start_time = time.time()
        capture_duration = 1800
        current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        out = cv2.VideoWriter(f"{outDir}{current_time}.mp4", fourcc, frameRate, frame_size)


        while( int(time.time() - start_time) < capture_duration ):
            
            _, frame = cap.read()  
              
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            datime = str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
                       
                       
            frame = cv2.putText(frame, datime, (7,235), font , 0.5 , (255,255,255), 1, cv2.LINE_AA)
            #Save clip
            out.write(frame)
            #Display video
            rframe = cv2.resize(frame, (width, height)) 
            cv2.imshow("CCTV", rframe)
         
            frame_time += 1
             
            if cv2.waitKey(1) == ord('q'):
                loop = False
                break
               
        out.release()
        clipNumber += 1
        
        print("Video: {clipNumber}, {current_time}")
    



CCTV()
cap.release()
cv2.destroyAllWindows()
