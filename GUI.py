import tkinter as tk
from PIL import ImageTk
import PIL.Image
from tkinter.ttk import *
from tkinter import filedialog
from tkinter import *
from PreData.PredataGUI import data_preparation, _model
from keras.models import load_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import librosa, librosa.display
import json

def choose_npz():
    global filename
    global data_X, data_y
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select file eeg",
                                          filetypes=(("npz files","*.npz"), ("all files", "*.*")))
    data_X, data_y = data_preparation(filename)
    spectrum_stft(data_X)
    print(">>>> done")
    return filename , data_X, data_y

def choose_npz_seq():
    global filename
    global data_X, data_X_seq
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select file eeg",
                                          filetypes=(("npz files","*.npz"), ("all files", "*.*")))
    data_X, data_X_seq = data_prepared_kfold(filename)
    return data_X, data_X_seq
    
def Load_model():
    global pre_model
    model = combo.get()
    if model == 'FPZ_CZ':
        pre_model = load_model('weights/eeg_fpz_cz/pre_model_10.h5')
        pre_model.summary()
        print(">>>> done")
    if model == 'PZ_OZ':
        pre_model = load_model('weights/eeg_pz_oz/pre_model_13.h5')
        pre_model.summary()
        print(">>>> done")
    else:   
        pass
    return pre_model

def predict():
    global y_pred, f1 , report , acc
    y_pred = pre_model.predict(data_X)
    y_pred = np.array([np.argmax(s) for s in y_pred])
    y_test = np.array([np.argmax(s) for s in data_y])
    f1 = f1_score(y_test, y_pred, average="macro")
    print(">>> f1 score: {}".format(f1))
    report = classification_report(y_test, y_pred)
    print(report)
    acc = accuracy_score(y_test, y_pred)
    print(">>> accucracy: {}".format(acc))
    fig = plt.figure(figsize=(20,8))
    plot = plt.subplot(111)
    plot.plot(y_test, label='test')
    plot.plot(y_pred, label='predict')
    plt.title('Sleep Scoring Base Single EEG signal')
    plt.xlabel("Time")
    plt.ylabel("Stage")
    plot.legend()
    plt.savefig('result.png')
    #plt.show()
    print(">>>> done")
    return y_pred, f1 , report , acc

def save_up_data():
    global json
    name_txt = textbox1.get()
    cmnd_txt = textbox2.get()
    sdt_txt = textbox3.get()
    y_pred_list = y_pred.tolist()
    #pass #save canvas
    file = {"name": name_txt, "cmnd": cmnd_txt, "sdt": sdt_txt, "sleep scoring": y_pred_list} 
    with open('database.json', 'w') as f:
        json.dump(file, f)
    print(">>>> done")
    #return json

def spectrum_stft(data_X):
    data_X = data_X.ravel()
    n_fft = 2048
    fs = 100
    hop_length = 512
    X = librosa.stft(data_X, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X))
    fig = plt.figure(figsize=(15, 5))
    librosa.display.specshow(S, sr=fs, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig('spectrum_stft.jpg')
    #print(">>>> done")

def show():
    
    global img, img2, img3
    img  = PIL.Image.open("result.png")
    img = img.resize((1000,360))
    img = ImageTk.PhotoImage(img)
    canvas1.create_image(450, 170, image=img)
    type_domain = combo1.get()
    if type_domain == 'Time&Frequence domain':
        img2 = PIL.Image.open("spectrum_stft.jpg")
        img2 = img2.resize((1000, 340))
        img2 = ImageTk.PhotoImage(img2)
        canvas2.create_image(450,150, image=img2)
    if type_domain == 'Time domain':
        X_ravel = data_X.ravel()
        #print(X_ravel)
        s_hour = tb1.get()
        s_minute = tb2.get()
        #print(hour, munite)
        s_time = int(s_hour)*60*60 + int(s_minute)*60 # seconds

        e_hour = tb3.get()
        e_minute = tb4.get()
        #print(hour, munite)
        e_time = int(e_hour)*60*60 + int(e_minute)*60 # seconds
        #print(time)
        #print(X_ravel.shape)
        print(X_ravel[s_time*100:e_time*100].shape) #Hz = 100
        pass
    else:
        pass
"""
    Graphical Use Interface Application Sleep Scoring 
"""

# Design Windows, panel 
root = tk.Tk()
root.title("APPLICATION STAGE SCORING")
root.geometry("1200x700")
root.resizable(0, 0)
root.configure(background='#808080')

panel = tk.Label(root,  borderwidth=2, relief="groove")
panel.place(x=20, y=200, width=250, height=85)

label_stft = tk.Label(root, text="Visualize signal",
                        bg='#f8f5fb', font=('times', 22),
                        width=57)
label_stft.place(x=290, y=351)


#Design text
#label_signal = tk.Label(root, text="Signal in time domain",
 #                       bg='#f8f5fb',font=('times', 22),
 #                       width= 28)
#label_signal.place(x=290, y=351)




label = tk.Label(panel, text="Tên:")
label.place(x=5, y=5, width=40)

label1 = tk.Label(panel, text= "CMND:")
label1.place(x=5, y=30, width=40)

label2 = tk.Label(panel, text="SÐT:")
label2.place(x=5, y=55, width=40)

label3 = tk.Label(root, text="Channel:", bg='#808080')
label3.place(x=20, y=400, width=47)

label7 = tk.Label(root, text="Domain:", bg='#808080')
label7.place(x=20, y=430, width=47)

############################################
panel_2 = tk.Label(root,  borderwidth=2, relief="groove")
panel_2.place(x=20, y=300, width=250, height=85)

label4 = tk.Label(panel_2, text="Choose Time Signal")
label4.place(x=50, y=5, width = 150)

label5 = tk.Label(panel_2, text="Start")
label5.place(x=5, y=30, width=40)

label6 = tk.Label(panel_2, text=":")
label6.place(x=125, y=30, width=40)

label8 = tk.Label(panel_2, text="End")
label8.place(x=5, y=55, width=40)

label9 = tk.Label(panel_2, text=":")
label9.place(x=125, y=55, width=40)



# Design Name Application
Name = tk.Label(root,
                 text="Application stage scoring\nbase on raw eeg signal",
                 pady=10,
                 bg='#FA8072',
                 font=('times', 18, 'italic'))
Name.place(anchor= tk.CENTER, x=143, y=45)
# Design Button
        #QUIT
but_quit = tk.Button(root, text="QUIT",
                     bg='#FA8072',
                   width=10, height=3,
                   justify=tk.LEFT,
                   command=root.destroy)
but_quit.place(x=150, y=635, width= 120)
        #PREDICT
but_predict = tk.Button(root, text="PREDICT",
                           width=10, height=3 ,
                           command=predict)
but_predict.place(x=20, y=570, width=120)
        #SAVE
but_save_data = tk.Button(root, text="SAVE DATA",
                           width=10, height=3,
                           command=save_up_data)
but_save_data.place(x=150, y=570, width=120)
        #SELECT
but_select = tk.Button(root, text="SELECT SIGNAL",
                   width=10, height=3,
                   command=choose_npz)
but_select.place(x=20, y=505, width=120)
        #LOAD MODEL
but_load_model = tk.Button(root, text="LOAD MODEL",
                           width=10, height=3,
                           command=Load_model)
but_load_model.place(x=150, y= 505, width= 120)
        #SHOW_RESULT
but_show = tk.Button(root, text="SHOW",
                     width=10, height=3,
                     command=show)
but_show.place(x=20, y= 635, width= 120)

#Design Canvas
        #CANVAS SLEEP STAGE
canvas1 = tk.Canvas(root, width=900, height=340, borderwidth=2, relief="groove")
canvas1.place(x=290, y=5)

        #CANVAS TIME DOMAIN
canvas2 = tk.Canvas(root, width=900, height=300, borderwidth=2, relief="groove")
canvas2.place(x=290, y=390)
        #CANVAS TIME-FREQUENCE DOMAIN
#canvas3 = tk.Canvas(root, width=445, height=300, borderwidth=2, relief="groove")
#canvas3.place(x=745, y=390)

# Design textbox
textbox1 = tk.Entry(panel)
textbox1.place(x=60, y=5, width=180)

textbox2 = tk.Entry(panel)
textbox2.place(x=60, y=30, width=180)

textbox3 = tk.Entry(panel)
textbox3.place(x=60, y=55, width=180)

combo = Combobox(root, value=('FPZ_CZ', 'PZ_OZ'))
combo.place(x=80, y=400, width=190)

tb1 = tk.Entry(panel_2)
tb1.place(x=60, y=30, width=50)

tb2 = tk.Entry(panel_2)
tb2.place(x=180, y=30, width=50)

tb3 = tk.Entry(panel_2)
tb3.place(x=60, y=55, width=50)

tb4 = tk.Entry(panel_2)
tb4.place(x=180, y=55, width=50)

combo1 = Combobox(root, value=('Time domain', 'Time&Frequence domain'))
combo1.place(x=80, y=430, width=190)



#Main
root.mainloop()
