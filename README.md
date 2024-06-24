# AI Camera - Nhận diện và Theo dõi Khuôn mặt

## Công nghệ sử dụng

- Framework Caffe với mô hình pre-trained `res10_300x300_ssd_iter_140000.caffemodel`
- Mô hình SVC (Support Vector Classifier) để phân loại các embedding khuôn mặt
- MTCNN từ FaceNet PyTorch để nhận dạng khuôn mặt
- Thuật toán SORT (Simple Online Realtime Tracking) để theo dõi khuôn mặt

## Cài đặt

1. Clone repository:
   ```
   git clone https://github.com/BaoToan1704/AI-Camera.git
   ```

2. Cài đặt các dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Clone repository SORT:
   ```
   git clone https://github.com/abewley/sort.git
   ```

4. Cài đặt dependencies cho SORT:
   ```
   cd sort
   pip install -r requirements.txt
   cd .. # hoặc cd - nếu sử dụng macOS
   ```

## Sử dụng

1. Trích xuất embedding khuôn mặt:
   ```
   python extract_embeddings.py
   ```

2. Huấn luyện mô hình:
   ```
   python train_model.py
   ```

3. Chạy chương trình nhận diện khuôn mặt:
   ```
   python nhandienkhuonmat.py
   ```

**Lưu ý:** Các file `extract_embeddings.py` và `train_model.py` chỉ cần chạy một lần nếu dataset không có sự thay đổi.