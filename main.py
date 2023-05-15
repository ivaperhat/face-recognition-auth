import numpy
import rsa
import face_recognition

def encryptEncodingsArray(face_encodings, public_key):
    encryptedArray = []

    for encoding in face_encodings:
        encoding = str(encoding)
        encrypted_encoding = rsa.encrypt(encoding.encode(), public_key)
        encryptedArray.append(encrypted_encoding)
    return encryptedArray # An encrypted array


def decryptEncodingsArray(encrypted_array, private_key):
    decryptedArray = []

    for encoding in encrypted_array:
        decEncodings = rsa.decrypt(encoding, private_key).decode()
        decryptedArray.append(decEncodings)

    decryptedArray = numpy.array(decryptedArray, dtype=numpy.float64)
    return decryptedArray # A numpy array of face encodings


# Generate the Keypair
publicKey, privateKey = rsa.newkeys(512)

# Generate and encrypt some face encodings
image = face_recognition.load_image_file("miley1.png")
face_encodings = face_recognition.face_encodings(image)[0]

# Encrypt face encodings using the generated public key
encrypted_face_encodings = encryptEncodingsArray(face_encodings, publicKey)

# Generate new face encodings
image = face_recognition.load_image_file("miley2.png")
new_face_encodings = face_recognition.face_encodings(image)[0]

# Use the private key to retrieve the encrypted face encodings
decrypted_face_encodings = decryptEncodingsArray(encrypted_face_encodings, privateKey)

results = face_recognition.compare_faces([decrypted_face_encodings], new_face_encodings)
face_distance = face_recognition.face_distance([decrypted_face_encodings], new_face_encodings)

print("Authentication will be successful if calculated face distance is 0.6 or less.")

if results[0] == 1:
    print(">>> AUTHENTICATION SUCCESSFULL! <<<")
else:
    print(">>> AUTHENTICATION FAILED. <<<")

print("Calculated face distance is: " + str(face_distance[0]))
