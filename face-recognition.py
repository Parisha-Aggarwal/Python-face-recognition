import face_recognition
import cv2
# Load known faces and their encodings
known_face_encodings = []
known_face_names = []
parisha_image = face_recognition.load_image_file("parisha.jpg")
parisha_encoding = face_recognition.face_encodings(parisha_image)[0]
ratan_tata_image = face_recognition.load_image_file("ratan.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]
andrew_image = face_recognition.load_image_file("andrew.jpg")
andrew_encoding = face_recognition.face_encodings(andrew_image)[0]
known_face_encodings.append(parisha_encoding)
known_face_names.append("Parisha Aggarwal")
known_face_encodings.append(ratan_tata_encoding)
known_face_names.append("Ratan Tata")
known_face_encodings.append(andrew_encoding)
known_face_names.append("Andrew Tate")
face_locations = []
face_encodings = []
face_names = []
# Open the webcam
video_capture = cv2.VideoCapture(0)
while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Resize frame to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Find all the face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
