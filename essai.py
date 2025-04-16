import cv2
import face_recognition

# Ouvrir la webcam
video_capture = cv2.VideoCapture(0)

# Charger une image de référence et l'encoder
known_image = face_recognition.load_image_file("Hery.jpg")  # Remplace par ton image de référence
known_encoding = face_recognition.face_encodings(known_image)[0]

# Nom associé à l'image de référence
known_names = ["Personne Connue"]

while True:
    # Capture une image de la webcam
    ret, frame = video_capture.read()

    # Convertir l'image en format RGB (face_recognition utilise RGB, OpenCV utilise BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détecter les visages dans l'image
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparer le visage détecté avec l'image de référence
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        name = "Inconnu"

        if True in matches:
            name = known_names[0]

        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Ajouter le nom sous le visage
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Afficher la vidéo en direct
    cv2.imshow('Reconnaissance Faciale', frame)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres
video_capture.release()
cv2.destroyAllWindows()
