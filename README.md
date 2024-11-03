# ReceiptOCR

# Trích Xuất Thông Tin Từ Hóa Đơn

Đây là dự án sử dụng các kỹ thuật xử lý ảnh, nhận dạng văn bản và phân loại thông tin để trích xuất dữ liệu từ hóa đơn. Dự án bao gồm các mô hình học sâu như DeepLabv3+, CNN và các hệ thống OCR (PaddleOCR, VietOCR) để giải quyết bài toán này.

Bạn có thể tham khảo `ReceiptOCR.ipynb` để chạy chương trình theo các bước dưới đây trên môi trường Google Colab.

## Mục lục
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Chạy chương trình](#chạy-chương-trình)
## Yêu cầu hệ thống

Trước khi bắt đầu, hãy đảm bảo rằng máy tính của bạn đáp ứng các yêu cầu sau:

- Python 3.7 trở lên
- pip (Python package installer)
- GPU
## Cài đặt
**Cài đặt các thư viện phụ thuộc**:

Tất cả các yêu cầu đã được liệt kê trong file `requirements.txt`. Bạn chỉ cần chạy lệnh sau để cài đặt:

```bash
pip install -r requirements.txt
```
Lưu ý, nếu chạy trên Google Colab với GPU cần phải thay đổi gói:
- torch==2.3.1+cu121 thành torch==2.3.1
- torchvision==0.18.1+cu121 thành torchvision==0.18.1

## Chạy chương trình

Sau khi cài đặt xong, bạn có thể chạy chương trình đơn giản như sau:

**Chạy end-to-end xử lý một hoặc nhiều ảnh hóa đơn**:

```bash
python main.py --image_path path/to/your/image
```

Trong đó, `path/to/your/image` là đường dẫn đến hình ảnh hóa đơn hoặc thư mục (không cần đuôi .jpg) bạn muốn xử lý.

