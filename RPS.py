import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
 
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense,MaxPool2D,Dropout,Flatten,Conv2D,GlobalAveragePooling2D,Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
from random import choice,shuffle
from scipy import stats as st
 
from collections import deque

from statistics import StatisticsError

def gather_data(num_samples):
     
    global rock, paper, scissor, nothing

    cap = cv2.VideoCapture(0)

    trigger = False

    counter = 0

    box_size = 234

    width = int(cap.get(3))
 
 
    while True:

        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        if not ret:
            break
  
        if counter == num_samples:
            trigger = not trigger
            counter = 0

        cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 150), 2)

        cv2.namedWindow("Collecting images", cv2.WINDOW_NORMAL)
         
 
        if trigger:

            roi = frame[5: box_size-5 , width-box_size + 5: width -5]

            eval(class_name).append([roi, class_name])
    
            counter += 1
   
            text = "Collected Samples of {}: {}".format(class_name, counter)
             
        else:
            text = "Press 'r' to collect rock samples, 'p' for paper, 's' for scissor and 'n' for nothing"

        cv2.putText(frame, text, (3, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Collecting images", frame)

        k = cv2.waitKey(1)

        if k == ord('r'):
 
            trigger = not trigger
            class_name = 'rock'
            rock = []
            
             
        if k == ord('p'):
            trigger = not trigger
            class_name = 'paper'
            paper = []
          
        if k == ord('s'):
            trigger = not trigger
            class_name = 'scissor'
            scissor = []
                     
        if k == ord('n'):
            trigger = not trigger
            class_name = 'nothing'
            nothing = []
         
        if k == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    return

no_of_samples = 100
gather_data(no_of_samples)

labels = [tupl[1] for tupl in rock] + [tupl[1] for tupl in paper] + [tupl[1] for tupl in scissor] +[tupl[1] for tupl in nothing]


images = [tupl[0] for tupl in rock] + [tupl[0] for tupl in paper] + [tupl[0] for tupl in scissor] +[tupl[0] for tupl in nothing]

images = np.array(images, dtype="float") / 255.0

print('Total images: {} , Total Labels: {}'.format(len(labels), len(images)))

encoder = LabelEncoder()

Int_labels = encoder.fit_transform(labels)


one_hot_labels = to_categorical(Int_labels, 4)

(trainX, testX, trainY, testY) = train_test_split(images, one_hot_labels, test_size=0.25, random_state=50)

images = []

image_size = 224

N_mobile = tf.keras.applications.NASNetMobile( input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')

N_mobile.trainable = False
    
x = N_mobile.output

x = GlobalAveragePooling2D()(x)

x = Dense(712, activation='relu')(x) 

x = Dropout(0.40)(x)

preds = Dense(4,activation='softmax')(x) 

model = Model(inputs=N_mobile.input, outputs=preds)

print ("Number of Layers in Model: {}".format(len(model.layers[:])))

augment = ImageDataGenerator( 
    
        rotation_range=30,
        zoom_range=0.25,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        horizontal_flip=False,    
        fill_mode="nearest"
)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 15
batchsize = 20

history = model.fit(x=augment.flow(trainX, trainY, batch_size=batchsize), validation_data=(testX, testY), 
steps_per_epoch= len(trainX) // batchsize, epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()

plt.show()

model.save("rps4.h5", overwrite=True)

model = load_model("rps4.h5")

label_names = ['nothing', 'paper', 'rock', 'scissor']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
box_size = 234
width = int(cap.get(3))

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
           
    cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 150), 2)
        
    cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)

    roi = frame[5: box_size-5 , width-box_size + 5: width -5]

    roi = np.array([roi]).astype('float64') / 255.0
 
    pred = model.predict(roi)

    target_index = np.argmax(pred[0])

    prob = np.max(pred[0])

    cv2.putText(frame, "prediction: {} {:.2f}%".format(label_names[np.argmax(pred[0])], prob*100 ),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Rock Paper Scissors", frame)
    
   
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def findout_winner(user_move, Computer_move):
    
    
    if user_move == Computer_move:
        return "Tie"
    
    
    elif user_move == "rock" and Computer_move == "scissor":
        return "User"
    
    elif user_move == "rock" and Computer_move == "paper":
        return "Computer"
    
    elif user_move == "scissor" and Computer_move == "rock":
        return "Computer"
    
    elif user_move == "scissor" and Computer_move == "paper":
        return "User"
    
    elif user_move == "paper" and Computer_move == "rock":
        return "User"
    
    elif user_move == "paper" and Computer_move == "scissor":
        return "Computer"
    
user_move = 'paper'
computer_move = choice(['rock', 'paper', 'scissor'])

winner = findout_winner(user_move, computer_move)

print("User Selected '{}' and computer selected '{}' , winner is: '{}' ".format(user_move, computer_move, winner))

def show_winner(user_score, computer_score):    
    
    if user_score > computer_score:
        img = cv2.imread("images/youwin.jpg")
        
    elif user_score < computer_score:
        img = cv2.imread("images/comwins.jpg")
        
    else:
        img = cv2.imread("images/draw.jpg")
        
    cv2.putText(img, "Press 'ENTER' to play again, else exit",
                (150, 530), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    
    cv2.imshow("Rock Paper Scissors", img)
    
    k = cv2.waitKey(0)
    
    if k == 13:
       return True

    else:
        return False
    
def display_computer_move(computer_move_name, frame):
    
        icon = cv2.imread( "images/{}.png".format(computer_move_name), 1)
        icon = cv2.resize(icon, (224,224))
        
        roi = frame[0:224, 0:224]

        mask = icon[:,:,-1] 

        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        icon_bgr = icon[:,:,:3] 
        
        
        img1_bg = cv2.bitwise_and(roi, roi, mask = cv2.bitwise_not(mask))

        img2_fg = cv2.bitwise_and(icon_bgr, icon_bgr, mask = mask)

        combined = cv2.add(img1_bg, img2_fg)

        frame[0:224, 0:224] = combined

        return frame



