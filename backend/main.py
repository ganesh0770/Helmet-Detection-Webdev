from fastapi import FastAPI, Form, Depends,UploadFile,File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import cv2 as cv
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from sqlalchemy.orm import Session
from datadb import SessionLocal, init_db, Contact
init_db()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
                   "http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# layers model

abels_value  = ['TN 09 BJ 4054', 'DL 14S C0887', 'MH38Q 6199', 'KA-09 HB-0164', 'AP13Q 8806', 'AP28 BA 7090', 'AP13Q 8815', 'AP13Q 8816', 'AP13Q 8780', 'HP37C 5788', 'AP3V 8078', 'AP13Q 1100', 'AP13Q 1121', 'AP11AV 2140', 'Ap13Q 1126', 'AP11AN 0591', 'AP13Q 1133', 'AP13Q 1136', 'AP13Q 1140', 'AP13Q 1177']    

plate_detecter = Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        input_shape=(64, 64, 3),
    ),

    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid"
    ),

    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid"
    ),

    Flatten(),

    Dense(
        units=128,
        activation="relu"
    ),

    Dense(
        units=20,
        activation="softmax"
    )
])

plate_detecter.load_weights("models/model_weights.h5")

classes = "Helmet"



global class_labels
global cnn_model
global cnn_layer_names


numberPlate=None
global option

inpWidth = 416       
inpHeight = 416  

frame_count = 0 
frame_count_out=0  

confThreshold = 0.5  
nmsThreshold = 0.4  



frame_count = 0 





# pretrained objec detection
modelConfiguration = "models/yolov3-obj.cfg";
modelWeights = "models/yolov3-obj_2400.weights";
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def labelsBoundingBoxes(image, Boundingbox, conf_thr, classID, ids, color_names, predicted_labels,indexno):
    option = 0
    if len(ids) > 0:
        for i in ids.flatten():
            # draw boxes
            xx, yy = Boundingbox[i][0], Boundingbox[i][1]
            width, height = Boundingbox[i][2], Boundingbox[i][3]

            class_color = (0,255,0)#[int(color) for color in color_names[classID[i]]]

            cv.rectangle(image, (xx, yy), (xx+width, yy+height), class_color, 2)
            print(classID[i])
            if classID[i] <= 1:
                text_label = "{}: {:4f}".format(predicted_labels[classID[i]], conf_thr[i])
                cv.putText(image, text_label, (xx, yy-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
                option = 1

    return image,option

def listBoundingBoxes(image, image_height, image_width, threshold_conf):
    box_array = []
    confidence_array = []
    class_ids_array = []

    for img in image:
        for obj_detection in img:
            detection_scores = obj_detection[5:]
            class_id = np.argmax(detection_scores)
            confidence_value = detection_scores[class_id]
            if confidence_value > threshold_conf and class_id <= 1:
                Boundbox = obj_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                center_X, center_Y, box_width, box_height = Boundbox.astype('int')

                xx = int(center_X - (box_width / 2))
                yy = int(center_Y - (box_height / 2))

                box_array.append([xx, yy, int(box_width), int(box_height)])
                confidence_array.append(float(confidence_value))
                class_ids_array.append(class_id)

    return box_array, confidence_array, class_ids_array

def detectObject(CNNnet, total_layer_names, image_height, image_width, image, name_colors, class_labels,indexno,  
            Boundingboxes=None, confidence_value=None, class_ids=None, ids=None, detect=True):

    if detect:
        blob_object = cv.dnn.blobFromImage(image,1/255.0,(416, 416),swapRB=True,crop=False)
        CNNnet.setInput(blob_object)
        cnn_outs_layer = CNNnet.forward(total_layer_names)
        Boundingboxes, confidence_value, class_ids = listBoundingBoxes(cnn_outs_layer, image_height, image_width, 0.5)
        ids = cv.dnn.NMSBoxes(Boundingboxes, confidence_value, 0.5, 0.3)
        if Boundingboxes is None or confidence_value is None or ids is None or class_ids is None:

            raise '[ERROR] unable to draw boxes.'
        image,option = labelsBoundingBoxes(image, Boundingboxes, confidence_value, class_ids, ids, name_colors, class_labels,indexno)

    return image,option





def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def loadLibraries(): #function to load yolov3 model weight and class labels
    global class_labels
    global cnn_model
    global cnn_layer_names
    class_labels = open('yolov3model/yolov3-labels').read().strip().split('\n') #reading labels from yolov3 model
    print(str(class_labels)+" == "+str(len(class_labels)))
    cnn_model = cv.dnn.readNetFromDarknet('yolov3model/yolov3.cfg', 'yolov3model/yolov3.weights') #reading model
    cnn_layer_names = cnn_model.getLayerNames() #getting layers from cnn model
    cnn_layer_names = [cnn_layer_names[i - 1] for i in cnn_model.getUnconnectedOutLayers()] #assigning all layers

def postprocess(frame, outs, option):
        global result_image
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        global frame_count_out
        frame_count_out=0
        classIds = []
        confidences = []
        boxes = []
        classIds = []
        confidences = []
        boxes = []
        cc = 0
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    #print(classIds)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        count_person=0 # for counting the classes in this loop.
        for i in indices:
            i = i
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height,frame,option)
            my_class='Helmet'      
            unknown_class = classes[classId]
            print("===="+str(unknown_class))
            if my_class == unknown_class:
                count_person += 1

        print(str(frame_count_out))
        if count_person == 0 and option == 1:
            cv.putText(frame, "Helmet Not detected", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        if count_person >= 1 and option == 0:
            #path = 'test_out/'
            #cv.imwrite(str(path)+str(cc)+".jpg", frame)     # writing to folder.
            #cc = cc + 1

            frame = cv.resize(frame,(500,500))
            result_image=frame


def drawPred(classId, conf, left, top, right, bottom,frame,option):
        global numberPlate
        global frame_count
        label = '%.2f' % conf
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        label_name,label_conf = label.split(':')
        Helmet.append(conf)
        print(label_name+" === "+str(conf)+"==  "+str(option))
        if label_name == 'Helmet' and conf > 0.50:
            if option == 0 and conf > 0.90:
                cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
                frame_count+=1

            if option == 0 and conf < 0.90:
                cv.putText(frame, "Helmet Not detected", (10, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                frame_count+=1
                img = cv.imread(filename)
                img = cv.resize(img, (64,64))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(1,64,64,3)
                X = np.asarray(im2arr)
                X = X.astype('float32')
                X = X/255
                preds = plate_detecter.predict(X)
                predict = np.argmax(preds)
                # img = cv.imread(filename)
                # img = cv.resize(img,(500,500))
                # text = tess.image_to_string(img, lang='eng')
                # text = text.replace("\n"," ")

                # messagebox.showinfo("Number Plate Detection Result", "Number plate detected as "+text)
                # textarea.insert(END,filename+"\n\n")
                # textarea.insert(END,"Number plate detected as "+str(labels_value[predict]))  
                # print("the number plate",str(labels_value[predict]))
                print("the number plate",numberPlate)
                # numberPlate=str(labels_value[predict])

            if option == 1:
                cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
                frame_count+=1   


        if(frame_count> 0):
            return frame_count
loadLibraries() 





def save_image(pil_image):
    os.makedirs("uploaded_images", exist_ok=True)
    base_name = os.path.splitext(pil_image.filename)[0]  # remove original extension
    pil_image.save(f"uploaded_images/{base_name}.png", format="PNG")


class Input_0(BaseModel):
    additional: str

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    Helmet=[]
    result_image=None
    file_bytes = await image.read()
    os.makedirs("uploads", exist_ok=True)
    with open(f"uploads/{image.filename}", "wb") as f:
        f.write(file_bytes)

    np_arr = np.frombuffer(file_bytes, np.uint8)
    filename = cv.imdecode(np_arr, cv.IMREAD_COLOR)

    if filename is None:
        raise Exception("Uploaded file is not a valid image.")


    def drawPred(classId, conf, left, top, right, bottom,frame,option):
        global numberPlate
        global frame_count
        #cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        label_name,label_conf = label.split(':')
        Helmet.append(conf)
        print(label_name+" === "+str(conf)+"==  "+str(option))
        if label_name == 'Helmet' and conf > 0.50:
            if option == 0 and conf > 0.90:
                cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
                frame_count+=1

            if option == 0 and conf < 0.90:
                cv.putText(frame, "Helmet Not detected", (10, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                frame_count+=1
                img = cv.imread(filename)
                img = cv.resize(img, (64,64))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(1,64,64,3)
                X = np.asarray(im2arr)
                X = X.astype('float32')
                X = X/255
                preds = plate_detecter.predict(X)
                predict = np.argmax(preds)
                # img = cv.imread(filename)
                # img = cv.resize(img,(500,500))
                # text = tess.image_to_string(img, lang='eng')
                # text = text.replace("\n"," ")

                # messagebox.showinfo("Number Plate Detection Result", "Number plate detected as "+text)
                # textarea.insert(END,filename+"\n\n")
                # textarea.insert(END,"Number plate detected as "+str(labels_value[predict]))  
                # print("the number plate",str(labels_value[predict]))
                print("the number plate",numberPlate)
                # numberPlate=str(labels_value[predict])

            if option == 1:
                cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
                frame_count+=1   


        if(frame_count> 0):
            return frame_count



    def postprocess(frame, outs, option):
        global result_image
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        global frame_count_out
        frame_count_out=0
        classIds = []
        confidences = []
        boxes = []
        classIds = []
        confidences = []
        boxes = []
        cc = 0
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    #print(classIds)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        count_person=0 # for counting the classes in this loop.
        for i in indices:
            i = i
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height,frame,option)
            my_class='Helmet'      
            unknown_class = classes[classId]
            print("===="+str(unknown_class))
            if my_class == unknown_class:
                count_person += 1

        print(str(frame_count_out))
        if count_person == 0 and option == 1:
            cv.putText(frame, "Helmet Not detected", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        if count_person >= 1 and option == 0:
            #path = 'test_out/'
            #cv.imwrite(str(path)+str(cc)+".jpg", frame)     # writing to folder.
            #cc = cc + 1

            frame = cv.resize(frame,(500,500))
            result_image=frame


    def detectBike():

        global option
        option = 0
        indexno = 0
        label_colors = (0,255,0)
        try:
            image=filename.copy()
            image_height, image_width = image.shape[:2]
        except:
            raise 'Invalid image path'
        finally:
            image, ops = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels,indexno)
            if ops == 1:
                option = 1





    def detectHelmet():
        if option == 1:
            frame= filename.copy()
            frame_count =0
            blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            postprocess(frame, outs,0)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            print("label",label)
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            print(label)
        else:
            print("Person & Motor bike not detected in uploaded image")




    
    detectBike()
    detectHelmet()

    Total_helmet=[]
    for value in Helmet:
        if value>=0.80:
            Total_helmet.append(value)
    if len(Total_helmet)>0:
        bike_detected=True
    else:
        bike_detected=False
 

    return {"helmet": bike_detected,"Total_helmet":len(Total_helmet),"result_image":result_image}
    


@app.post("/predict_video")
async def predict(video: UploadFile = File(...)):
    Helmet=[]
    result_image=None
    def drawPred(classId, conf, left, top, right, bottom,frame,option):
        global numberPlate
        global frame_count
        #cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        label_name,label_conf = label.split(':')
        Helmet.append(conf)
        print(label_name+" === "+str(conf)+"==  "+str(option))
        if label_name == 'Helmet' and conf > 0.50:
            if option == 0 and conf > 0.90:
                cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
                frame_count+=1

            if option == 0 and conf < 0.90:
                cv.putText(frame, "Helmet Not detected", (10, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                frame_count+=1
                img = cv.imread(filename)
                img = cv.resize(img, (64,64))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(1,64,64,3)
                X = np.asarray(im2arr)
                X = X.astype('float32')
                X = X/255
                preds = plate_detecter.predict(X)
                predict = np.argmax(preds)
                # img = cv.imread(filename)
                # img = cv.resize(img,(500,500))
                # text = tess.image_to_string(img, lang='eng')
                # text = text.replace("\n"," ")

                # messagebox.showinfo("Number Plate Detection Result", "Number plate detected as "+text)
                # textarea.insert(END,filename+"\n\n")
                # textarea.insert(END,"Number plate detected as "+str(labels_value[predict]))  
                # print("the number plate",str(labels_value[predict]))
                print("the number plate",numberPlate)
                # numberPlate=str(labels_value[predict])

            if option == 1:
                cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
                frame_count+=1   


        if(frame_count> 0):
            return frame_count
    def postprocess(frame, outs, option):
        global result_image
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        global frame_count_out
        frame_count_out=0
        classIds = []
        confidences = []
        boxes = []
        classIds = []
        confidences = []
        boxes = []
        cc = 0
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    #print(classIds)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        count_person=0 # for counting the classes in this loop.
        for i in indices:
            i = i
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height,frame,option)
            my_class='Helmet'      
            unknown_class = classes[classId]
            print("===="+str(unknown_class))
            if my_class == unknown_class:
                count_person += 1

        print(str(frame_count_out))
        if count_person == 0 and option == 1:
            cv.putText(frame, "Helmet Not detected", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        if count_person >= 1 and option == 0:
            #path = 'test_out/'
            #cv.imwrite(str(path)+str(cc)+".jpg", frame)     # writing to folder.
            #cc = cc + 1

            frame = cv.resize(frame,(500,500))
            result_image=frame

    os.makedirs("uploads", exist_ok=True)

    video_path = f"uploads_videos/{video.filename}"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    video = cv.VideoCapture(video_path)
    frame_id = 0
    fps = video.get(cv.CAP_PROP_FPS)
    desired_fps = 3  # 2 frames per second
    # frame_interval = max(1, int(fps / desired_fps))
    frame_interval=45
    while(True):
        ret, frame = video.read()
        if frame_id % frame_interval != 0:
            frame_id += 1
            continue
        if ret == True:
            frame_count = 0
            filename = "temp.png"
            cv.imwrite("temp.png",frame)
            blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            postprocess(frame, outs,1)
            t, _ = net.getPerfProfile()
 
            frame_id += 1
        else:
            break
    video.release()
    cv.destroyAllWindows()
    
    if len(Helmet)>0:
        bike_detected=True
    else:   
        bike_detected=False

    print("the total helmet in video",len(Helmet))
 

    return {"helmet": bike_detected,"Total_helmet":len(Helmet),"result_image":result_image}







    


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/contact_data")
def contact_data(
    name: str = Form(...),
    email: str = Form(...),
    text_box: str = Form(...),
    db: Session = Depends(get_db)
):

    # print("Name:", name)
    # print("Email:", email)
    # print("Message:", text_box)

    new_contact = Contact(name=name, email=email, text_box=text_box)
    db.add(new_contact)
    db.commit()
    db.refresh(new_contact)

    # print(f"Saved contact: {new_contact.name}, {new_contact.email}, {new_contact.text_box}")

    return True

