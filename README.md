# IMAGE_RECOGNITION-WITH-TENSORFLOW
A simple project on Image recognition on flowers using tensorflow

# 🌸 Flower Image Classification API & Frontend

This project is a **flower image classification system** built using **TensorFlow**, **FastAPI**, **Docker**, and **Streamlit**. The backend model classifies images of flowers into one of five classes: `daisy`, `dandelion`, `roses`, `sunflowers`, or `tulips`.

It supports:

- ✅ A RESTful API for predictions (FastAPI)
- ✅ A user-friendly interface (Streamlit)
- ✅ Full containerization (Docker)
- ✅ Deployment on the cloud (Render)

---

## 🧠 Model Overview

The model is a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It includes **BatchNormalization**, pooling, and fully connected layers.

It is trained and saved using:

```python
model = create_model(num_classes)
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy(from_logits=False)', metrics=['accuracy'])
model.save("mobilenetv2.h5")
```

---

## 🧪 Prediction API (FastAPI)

### ✅ Endpoints

| Method | Path       | Description                 |
|--------|------------|-----------------------------|
| GET    | `/`        | Health check                |
| POST   | `/predict/`| Upload an image for prediction |

### ✅ Example Response

```json
{
  "prediction": "daisy",
  "confidence": 0.945
}
```

---

## 🎨 Streamlit Frontend

The Streamlit app allows users to:

- Upload an image of a flower
- Send the image to the FastAPI backend
- Display prediction result and confidence score

---

## 🐳 Dockerized App

### ✅ Dockerfile

```dockerfile
FROM tensorflow/tensorflow:2.15.0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
```

---

## 🚀 Deployment on Render

### ✅ render.yaml

```yaml
services:
  - type: web
    name: flower-fastapi-app
    env: docker
    plan: free
    region: oregon
    dockerfilePath: ./Dockerfile
    autoDeploy: true
    envVars:
      - key: PORT
        value: 8000
```

### 🖥 Deploying Streamlit Frontend

You can also deploy the `frontend/streamlit_app.py` as a separate web service using Python environment:

- **Runtime**: Python
- **Start Command**:  
  ```bash
  streamlit run frontend/streamlit_app.py --server.port $PORT --server.enableCORS false
  ```

---

## 📁 Project Structure

```
.
├── app.py                       # FastAPI backend
├── mobilenetv2.h5               # Trained TensorFlow model
├── Dockerfile                   # Docker setup
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render deployment config
└── frontend/
    └── streamlit_app.py         # Streamlit UI
```

---

## 📦 Install & Run Locally

### 🔹 Backend (FastAPI)

```bash
# Clone the repo
git clone https://github.com/your-username/flower-classifier.git
cd flower-classifier

# Build Docker image
docker build -t flower-api .

# Run container
docker run -p 8000:8000 flower-api
```

### 🔹 Frontend (Streamlit)

```bash
cd frontend
streamlit run streamlit_app.py
```

---

## ✨ Features

- Lightweight and fast API
- Clean and interactive user interface
- Dockerized for portability
- Easily customizable for other classification tasks

---

## 🛠 Tech Stack

- TensorFlow / Keras
- FastAPI
- Docker
- Streamlit
- Render (for cloud deployment)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Render Cloud](https://render.com/)

---

## 💬 Contact

Built by **[Clifford Ojuka]**  
🔗 GitHub: [https://github.com/Clffordojuka]  
📫 Reach out for collaborations or questions!

---