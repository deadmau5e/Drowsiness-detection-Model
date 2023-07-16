import tensorflow as tf
import cv2 
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random  # Add this line to import the random module
from tensorflow import keras
from tensorflow.keras import layers

folder = 'train'
path = "C:\\Users\\gajgh\\Documents\\IITB\\coding\\SUMMER\\SOC\\archive\\train"

classes = ['Open_Eyes', 'Closed_Eyes']

train_data = []

for i in classes:
    path = os.path.join("C:\\Users\\gajgh\\Documents\\IITB\\coding\\SUMMER\\SOC\\archive", folder, i)
    class_num = classes.index(i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        new_array = cv2.resize(rgb, (224, 224))
        train_data.append([new_array, class_num])

random.shuffle(train_data)

x = []
y = []

for i, j in train_data:
    x.append(i)
    y.append(j)

x = np.array(x).reshape(-1, 224, 224, 3)
x = x / 255.0
y = np.array(y)

pickle_out = open('x.pickle', 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open('x.pickle', 'rb')
x = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
pickle_in.close()


model=tf.keras.applications.mobilenet.MobileNet()

model.summary()

base_input=model.layers[0].input

base_output=model.layers[-4].output

flat_layer=layers.Flatten()(base_output)
final_output=layers.Dense(1)(flat_layer)
final_op=layers.Activation('sigmoid')(final_output)

new_model=keras.Model(inputs=base_input,outputs=final_output)

new_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

new_model.fit(x,y,epochs=20,validation_split=0.1)

new_model.save('my_model.h5')

new_model=tf.keras.models.load_model('my_model.h5')

img=cv2.imread('open-test_1.png',cv2.IMREAD_GRAYSCALE)
rgb=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
new=cv2.resize(rgb,(224,224))

x_input=np.array(new).reshape(1,224,224,3)

x_input.shape

x_input=x_input/255.0

plt.imshow(new)

prediction=new_model.predict(x_input)

prediction

img=cv2.imread('full_face.jpg')

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_eye.xml')

gray=cv2.cvtColor(img, cv2.COLO_BGR2GRAY)

eyes=eye_cascade.detectMultiScale(gray,1.1,4)

for x,y,w,h in eyes[:2]:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    eyes=eye_cascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in eyes[:2]:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyess=eye_cascade.detectMultiScale(roi_gray)
        if len(eyess)==0:
            print('eyes are not detected')
        else:
            for (ex,ey,ew,eh) in eyess:
                eyes_roi=roi_color[ey:ey+eh,ex:ex+ew]

        plt.imshow(cv2.cvtColor(eyes_roi,cv2.COLOR_BGR2RGB))

        final_image=cv2.resize(eyes_roi,(224,224))
        final_image=np.expand_dims(final_image,axis=0)
        final_image=final_image/255.0

        final_image.shape

        predictions=new_model.predict(final_image)

        predictions

        cap=cv2.VideoCapture(0)
        if not cap.isOpened():
            cap=cv2.VideoCapture(1)
        if not cap.isOpened():
            raise IOError('cannot open webcam')

        while True:
                cbs=0

                success, frame=cap.read()
                if not success:
                    break
                else:
                    face_casade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
                    eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
                    faces=face_cascade.detectMultiScale(frame,1.1,7)
                    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    for(x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y)(x+w,y+h),(255,0,0),2)
                        roi_gray=gray[y:y+h,x:x+w]
                        roi_color=frame[y:y+h,x:x+w]
                        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,3)
                        if len(eyes)==0:
                            print('Eyes not detected')
                            cbs=0
                            break
                        
                    for (ex,ey,ew,eh) in eyes:
                        cbs=1
                        cv2.rectangle(roi_color, (ex,ey),
                                      (ex+ew,ey+eh),(0,255,0),2)
                        eyes_roi=roi_color[ey:ey+eh,ex:ex+ew]

                        if cbs==1:
                            final_image=cv2.resize(eyes_roi,(224,224))
                            final_image=np.expand_dims(final_image, axis=0)
                            final_image=final_image/255.0
                            predictions=new_model.predict(final_image)
                        if (predictions[0][0]<0):
                            status="Open Eyes"
                        else:
                            status="Closed Eyes"
                        font=cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,status,(50,50),font,
                                    3,(0,0,255),2,cv2.LINE_4)
                    cv2.imshow('Driver Drowsiness detection',frame)
                    
                    if cv2.waitKey(2) & 0xFF==ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()



