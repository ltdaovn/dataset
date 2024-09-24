## Hướng dẫn xây dựng trang web dự đoán loại hoa Iris sử dụng Flask và Decision Tree

**Mục tiêu:** Xây dựng một trang web đơn giản sử dụng Flask để dự đoán loại hoa Iris dựa trên các thông số người dùng nhập vào. Mô hình dự đoán được xây dựng dựa trên thuật toán Decision Tree và được lưu lại để sử dụng sau này.

**Bước 1: Huấn luyện mô hình và lưu trữ**

1. **Chuẩn bị môi trường:**
    - Cài đặt Python và thư viện cần thiết:
        ```bash
        pip install pandas scikit-learn flask
        ```
    - Tải tập dữ liệu Iris từ [https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris) và lưu vào thư mục dự án.
2. **Đọc dữ liệu:**
    - Sử dụng thư viện Pandas để đọc dữ liệu từ file CSV:
        ```python
        import pandas as pd

        data = pd.read_csv('iris.csv')
        ```
3. **Chuẩn bị dữ liệu:**
    - Chia dữ liệu thành tập huấn luyện và tập kiểm tra:
        ```python
        from sklearn.model_selection import train_test_split

        X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = data['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ```
4. **Huấn luyện mô hình Decision Tree:**
    - Khởi tạo mô hình Decision Tree:
        ```python
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier()
        ```
    - Huấn luyện mô hình:
        ```python
        model.fit(X_train, y_train)
        ```
    - Đánh giá mô hình: (sinh viên tự thực hiện)
5. **Lưu mô hình đã huấn luyện:**
    - Sử dụng thư viện `pickle` để lưu mô hình:
        ```python
        import pickle

        filename = 'iris_model.pkl'
        pickle.dump(model, open(filename, 'wb'))
        ```

**Bước 2: Xây dựng giao diện web và dự đoán**

1. **Khởi tạo ứng dụng Flask:**
    - Tạo file `app.py` và thêm các import cần thiết:
        ```python
        from flask import Flask, render_template, request
        import pickle

        app = Flask(__name__)
        ```
2. **Load mô hình đã huấn luyện:**
    - Load mô hình từ file đã lưu:
        ```python
        filename = 'iris_model.pkl'
        model = pickle.load(open(filename, 'rb'))
        ```
3. **Tạo trang web:**
    - Tạo trang web `index.html` với form nhập liệu:
        ```html
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Dự đoán loại hoa Iris</title>
        </head>
        <body>
            <h1>Dự đoán loại hoa Iris</h1>
            <form method="POST">
                <label for="sepal_length">Chiều dài đài hoa:</label>
                <input type="number" name="sepal_length" id="sepal_length" required><br><br>
                <label for="sepal_width">Chiều rộng đài hoa:</label>
                <input type="number" name="sepal_width" id="sepal_width" required><br><br>
                <label for="petal_length">Chiều dài cánh hoa:</label>
                <input type="number" name="petal_length" id="petal_length" required><br><br>
                <label for="petal_width">Chiều rộng cánh hoa:</label>
                <input type="number" name="petal_width" id="petal_width" required><br><br>
                <button type="submit">Dự đoán</button>
            </form>
            {% if prediction %}
                <h2>Kết quả: {{ prediction }}</h2>
            {% endif %}
        </body>
        </html>
        ```
4. **Xử lý dữ liệu người dùng:**
    - Xây dựng route xử lý dữ liệu từ form:
        ```python
        @app.route('/', methods=['GET', 'POST'])
        def index():
            prediction = None
            if request.method == 'POST':
                sepal_length = float(request.form['sepal_length'])
                sepal_width = float(request.form['sepal_width'])
                petal_length = float(request.form['petal_length'])
                petal_width = float(request.form['petal_width'])

                input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
                prediction = model.predict(input_data)[0]

            return render_template('index.html', prediction=prediction)
        ```
5. **Khởi chạy ứng dụng:**
    - Thực hiện lệnh sau trong terminal:
        ```bash
        flask run
        ```
    - Truy cập vào địa chỉ được hiển thị trong terminal để sử dụng trang web.

**Lưu ý:**
- Thay đổi `iris.csv` bằng đường dẫn file dữ liệu Iris của bạn.
- Thay đổi `filename` bằng tên file lưu mô hình.
- Điều chỉnh giao diện web theo ý muốn.
