import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# VGG16 모델 정의
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 마지막 4개 레이어를 trainable로 설정
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 이미지 전처리 및 데이터 증강
def preprocess_data(train_dir, valid_dir):
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
    valid_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224),
                                                        batch_size=32, class_mode='binary')
    valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(224, 224),
                                                        batch_size=32, class_mode='binary')
    return train_generator, valid_generator

# 비디오에서 프레임 추출
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

# 모델 학습
def train_model():
    train_dir = 'C:/Users/yywme/OneDrive/바탕 화면/dataset/train_data'
    valid_dir = 'C:/Users/yywme/OneDrive/바탕 화면/dataset/valid_data'

    model = create_vgg16_model()
    train_generator, valid_generator = preprocess_data(train_dir, valid_dir)

    # EarlyStopping 및 ReduceLROnPlateau 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # 모델 학습
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=valid_generator,
        callbacks=[early_stopping, reduce_lr]  # 콜백 추가
    )

    model.save('deepfake1_detector_model.keras')  # 모델 저장
    plot_training_history(history)  # 학습 결과 시각화

# 학습 결과 시각화
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# 파일 선택 및 딥페이크 판별
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("모든 파일", "*.*"), ("MP4 files", "*.mp4"), ("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png")])
    if file_path:
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            output_folder = 'C:/Users/yywme/OneDrive/바탕 화면/dataset/deepfake_frames'
            extract_frames(file_path, output_folder)

            total_frames, deepfake_frames = process_video(file_path)
            deepfake_percentage = (deepfake_frames / total_frames) * 100 if total_frames > 0 else 0

            result_message = f"총 {total_frames} 프레임 중 {deepfake_frames} 프레임이 딥페이크로 판별되었습니다.\n"
            result_message += f"딥페이크 확률: {deepfake_percentage:.2f}%"
            
            messagebox.showinfo("딥페이크 판별 결과", result_message)
            show_video_preview(file_path)
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            frame = cv2.imread(file_path)
            if frame is None:
                messagebox.showerror("오류", "이미지를 읽을 수 없습니다. 파일 경로를 확인하세요.")
                return
            
            if detect_deepfake(frame):
                messagebox.showinfo("딥페이크 판별 결과", "이 이미지는 딥페이크입니다.")
            else:
                messagebox.showinfo("딥페이크 판별 결과", "이 이미지는 진짜입니다.")

def detect_deepfake(frame):
    processed_frame = preprocess_frame(frame)
    model = load_model('deepfake1_detector_model.keras')
    prediction = model.predict(processed_frame)
    print(f"Prediction: {prediction[0]}")  # 예측 결과 출력
    return prediction[0] > 0.5

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    deepfake_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if detect_deepfake(frame):
            deepfake_count += 1
    
    cap.release()
    return frame_count, deepfake_count

def show_video_preview(video_path):
    cap = cv2.VideoCapture(video_path)
    
    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(img)
            label.config(image=img)
            label.image = img
            label.after(10, update_frame)
        else:
            cap.release()

    window = tk.Toplevel()
    window.title("비디오 미리보기")
    label = tk.Label(window)
    label.pack()
    
    update_frame()
    window.mainloop()

# 학습 및 판별을 위한 GUI 설정
root = tk.Tk()
root.title("딥페이크 감지 앱")
root.geometry("600x400")

# 학습 버튼
btn_train_model = tk.Button(root, text="모델 학습", command=train_model, font=("Arial", 14))
btn_train_model.pack(pady=10)

# 파일 선택 버튼
btn_select_file = tk.Button(root, text="파일 선택", command=select_file, font=("Arial", 14))
btn_select_file.pack(pady=10)

# 종료 버튼
exit_button = tk.Button(root, text="종료", font=("Helvetica", 12), bg="#f44336", fg="white", command=root.quit, relief="flat", width=15)
exit_button.pack(pady=10)

root.mainloop()
