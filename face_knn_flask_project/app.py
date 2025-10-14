import os, sqlite3, io, base64, shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
import cv2, numpy as np, joblib
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(MODEL_DIR, "face_knn.db")
MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.joblib")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(HAAR_PATH)

IMG_W = 100
IMG_H = 100
CAPTURE_COUNT = 15  # number of images to capture per user via web

app = Flask(__name__)

# --- Database helpers ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre TEXT NOT NULL,
                    carpeta TEXT NOT NULL,
                    fecha_registro TEXT NOT NULL
                )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS accesos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario_id INTEGER,
                    nombre TEXT,
                    fecha_hora TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# --- Utilities ---
def ensure_user_folder(uid):
    folder = os.path.join(DATA_DIR, str(uid))
    os.makedirs(folder, exist_ok=True)
    return folder

def preprocess_face_from_bgr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (IMG_W, IMG_H))
    return face.flatten()

def detect_and_crop(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    if len(faces) == 0:
        return None
    x,y,w,h = faces[0]
    pad = 10
    y1 = max(0, y-pad); y2 = min(image_bgr.shape[0], y+h+pad)
    x1 = max(0, x-pad); x2 = min(image_bgr.shape[1], x+w+pad)
    return image_bgr[y1:y2, x1:x2]

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            return jsonify({"success": False, "error": "Nombre requerido"}), 400
        conn = get_db_connection()
        cur = conn.cursor()
        fecha = datetime.now().isoformat(sep=" ", timespec="seconds")
        cur.execute("INSERT INTO usuarios (nombre, carpeta, fecha_registro) VALUES (?, ?, ?)", (name, "", fecha))
        uid = cur.lastrowid
        carpeta = str(uid)
        cur.execute("UPDATE usuarios SET carpeta = ? WHERE id = ?", (carpeta, uid))
        conn.commit()
        conn.close()
        ensure_user_folder(uid)
        return jsonify({"success": True, "user_id": uid})
    else:
        return render_template("register.html")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    user_id = request.form.get("user_id", None)
    image_data = request.form.get("image_data", None)
    if user_id is None or image_data is None:
        return jsonify({"success": False, "error": "Faltan parámetros"}), 400
    try:
        uid = int(user_id)
    except:
        return jsonify({"success": False, "error": "user_id inválido"}), 400
    header, encoded = image_data.split(",", 1)
    data = base64.b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Imagen no decodificable"}), 400
    cropped = detect_and_crop(img)
    folder = ensure_user_folder(uid)
    count = len([n for n in os.listdir(folder) if n.lower().endswith((".jpg",".png"))])
    filename = os.path.join(folder, f"user_{uid}_{count+1}.jpg")
    if cropped is not None:
        cv2.imwrite(filename, cropped)
    else:
        resized = cv2.resize(img, (IMG_W, IMG_H))
        cv2.imwrite(filename, resized)
    return jsonify({"success": True, "filename": os.path.basename(filename)})

@app.route("/train", methods=["POST","GET"])
def train():
    X = []
    y = []
    conn = get_db_connection()
    rows = conn.execute("SELECT id, nombre FROM usuarios").fetchall()
    for row in rows:
        uid = row["id"]
        folder = os.path.join(DATA_DIR, str(uid))
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg",".png",".jpeg")):
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                if img is None:
                    continue
                cropped = detect_and_crop(img)
                if cropped is None:
                    face = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (IMG_W, IMG_H))
                else:
                    face = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), (IMG_W, IMG_H))
                X.append(face.flatten())
                y.append(uid)
    conn.close()
    if len(X) == 0:
        return jsonify({"success": False, "error": "No hay imágenes para entrenar"}), 400
    X = np.array(X)
    y = np.array(y)
    knn = KNeighborsClassifier(n_neighbors=min(3, len(y)), metric="euclidean")
    knn.fit(X, y)
    joblib.dump(knn, MODEL_PATH)
    return jsonify({"success": True, "samples": int(len(y)), "model": os.path.basename(MODEL_PATH)})

@app.route("/recognize", methods=["POST","GET"])
def recognize():
    if request.method == "GET":
        return render_template("recognize.html")
    image_data = request.form.get("image_data", None)
    if image_data is None:
        return jsonify({"success": False, "error": "image_data requerido"}), 400
    if not os.path.exists(MODEL_PATH):
        return jsonify({"success": False, "error": "Modelo no entrenado"}), 400
    header, encoded = image_data.split(",", 1)
    data = base64.b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Imagen inválida"}), 400
    cropped = detect_and_crop(img)
    if cropped is None:
        face = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (IMG_W, IMG_H))
    else:
        face = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), (IMG_W, IMG_H))
    vec = face.flatten().reshape(1, -1)
    knn = joblib.load(MODEL_PATH)
    pred = int(knn.predict(vec)[0])
    neigh_dist, neigh_idx = knn.kneighbors(vec, n_neighbors=min(3, len(knn._fit_X)), return_distance=True)
    avg_dist = float(neigh_dist.mean())
    label = "Desconocido" if avg_dist > 3000 else None
    conn = get_db_connection()
    if label is None:
        row = conn.execute("SELECT nombre FROM usuarios WHERE id = ?", (pred,)).fetchone()
        if row:
            label = row["nombre"]
            conn.execute("INSERT INTO accesos (usuario_id, nombre, fecha_hora) VALUES (?, ?, ?)", (pred, label, datetime.now().isoformat(sep=' ', timespec='seconds')) )
            conn.commit()
    conn.close()
    return jsonify({"success": True, "pred": int(pred), "label": label, "avg_dist": avg_dist})

@app.route("/logs")
def logs():
    conn = get_db_connection()
    rows = conn.execute("SELECT id, usuario_id, nombre, fecha_hora FROM accesos ORDER BY fecha_hora DESC").fetchall()
    conn.close()
    return render_template("logs.html", rows=rows)

@app.route("/reset_logs", methods=["POST"])
def reset_logs():
    conn = get_db_connection()
    conn.execute("DELETE FROM accesos")
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# NUEVA FUNCIÓN: eliminar un registro específico
@app.route("/delete_log/<int:log_id>", methods=["POST"])
def delete_log(log_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM accesos WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "deleted_id": log_id})

@app.route("/delete_all", methods=["POST"])
def delete_all():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    init_db()
    return jsonify({"success": True})

@app.route("/users")
def users():
    conn = get_db_connection()
    rows = conn.execute("SELECT id, nombre, carpeta, fecha_registro FROM usuarios").fetchall()
    conn.close()
    data = [dict(r) for r in rows]
    return jsonify({"users": data})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
