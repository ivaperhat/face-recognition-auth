from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_lfw_pairs
import tqdm
import os
import matplotlib.pyplot as plt
import face_recognition

lfw = fetch_lfw_pairs(subset='test', color=True, resize=1)

pairs = lfw.pairs
labels = lfw.target
predictions = []
actuals = []


# Get predictions for LFW pairs using face_match() function
def get_predictions():
    for idx in tqdm.tqdm(range(0, pairs.shape[0])):
        pair = pairs[idx]
        img1 = pair[0]
        img2 = pair[1]

        plt.imshow(img1 / 255)
        plt.savefig('fig1.jpg')
        plt.imshow(img2 / 255)
        plt.savefig('fig2.jpg')

        fig1_array = face_recognition.load_image_file('fig1.jpg')
        fig2_array = face_recognition.load_image_file('fig2.jpg')

        face_encodings1 = face_recognition.face_encodings(fig1_array)
        face_encodings2 = face_recognition.face_encodings(fig2_array)

        if len(face_encodings1) == 0 or len(face_encodings2) == 0: # If a face is not detected in one of the images
            prediction = False
        else:
            result = face_recognition.compare_faces(face_encodings1, face_encodings2[0])[0]
            if result:
                prediction = True
            else:
                prediction = False

        predictions.append(prediction)

    # Remove figure files
    os.remove('fig1.jpg')
    os.remove('fig2.jpg')

    return predictions


# Compare lists to print confusion matrix
def get_confusion_matrix(actuals_list, predictions_list):
    cm = confusion_matrix(actuals_list, predictions_list)
    tn, fp, fn, tp = cm.ravel()

    return tp, tn, fp, fn


# Test for: accuracy, precision, recall, f1
def calaulate_metrics(actuals_list, predictions_list):
    accuracy = 100 * accuracy_score(actuals_list, predictions_list)
    precision = 100 * precision_score(actuals_list, predictions_list)
    recall = 100 * recall_score(actuals_list, predictions_list)
    f1 = 100 * f1_score(actuals_list, predictions_list)

    return accuracy, precision, recall, f1


actuals_list = labels
predictions_list = get_predictions()

# Print Test Results
print(get_confusion_matrix(actuals_list, predictions_list))
print(calaulate_metrics(actuals_list, predictions_list))
