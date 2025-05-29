# Computer Vision

Nhóm 7. Đề tài: Nhận Diện Các Loại Bệnh Thường Gặp Trên Ngô Thông Qua Lá Cây

Notebook nên được import trên kaggle để chạy
## 1. Hướng dẫn tải dữ liệu
- Truy cập "https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset" để tải dữ liệu. Nếu chạy trên kaggle thì vào "Add input" và dán link dữ liệu vào

## 2. Trainning
- Nếu chạy trên kaggle thì vào Settings -> Accelerator -> GPU T4 x2. Sau đó chạy code
- Nếu chạy ở máy local thì cần thay base_dir, output_dir và data_dir thành đường dẫn dữ liệu đã tải. Sau đó chạy code
## 3. Inference

Cách sử dụng mô hình đã huấn luyện (plant_disease_classifier.h5) để dự đoán trên dữ liệu mới:

- Load lại mô hình đã lưu
```bash
model = load_model('plant_disease_classifier.h5')
```
- Tiền xử lý ảnh mới
```bash
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
```
- Dự đoán nhãn cho ảnh mới
```bash
labels = ['chay-la', 'dom-la', 'gi-sat', 'khoe-manh']
img_array = preprocess_image('duong_dan_anh_du_doan.jpg') # sửa lại đường dẫn tới ảnh cần dự đoán
preds = model.predict(img_array)
pred_class = np.argmax(preds)
print(f'Ảnh dự đoán là: {labels[pred_class]} (score: {preds[0][pred_class]:.2f})')
```
