
# Face KNN - Web Version (Flask)

Instrucciones rápidas:
1. Crear y activar un entorno virtual (recomendado):
   python -m venv venv
   venv\Scripts\activate   (Windows)   OR   source venv/bin/activate  (Linux/Mac)

2. Instalar dependencias:
   pip install -r requirements.txt

3. Ejecutar la aplicación:
   python app.py

4. Abrir en el navegador:
   http://localhost:5000/

Flujos principales:
- Registrar usuario: captura desde la cámara (captura ~15 imágenes por usuario).
- Entrenar modelo: desde la página principal, entrena KNN usando imágenes guardadas.
- Reconocer: captura una imagen y la envía al servidor; si es reconocido, queda registrado en la base de datos (model/face_knn.db).
- Ver registros: ver el historial de accesos.
- Reiniciar registros / Eliminar TODO: operaciones administrativas.
