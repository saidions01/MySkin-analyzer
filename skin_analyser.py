import cv2
from ollama import chat
from PIL import Image
import base64
import io
import os

def encode_image(image_path):
    img = Image.open(image_path)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def capture_from_webcam(output_path="captured_image.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open webcam")

    print("Press 's' to capture the image...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam - Press s to save', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(output_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_path

def analyze_skin(image_path):
    image_base64 = encode_image(image_path)

    prompt = """
    Analyze the attached face image carefully and describe the following:
    1. Skin type (oily, combination, dry, sensitive, etc.)
    2. Any visible skin concerns such as acne, redness, dryness, hyperpigmentation, wrinkles, pores, dark circles, etc.
    3. The overall tone and condition of the skin.

    Then, based on your analysis, recommend:
    - A 3-step morning routine (with product examples)
    - A 3-step night routine
    - A weekly treatment if needed (e.g., exfoliation or mask)
    - Lifestyle and diet tips

    Respond in a friendly and empathetic tone. Your output should be tailored to this person and not a generic list.
    """


    response = chat(
    model="llava-phi3:latest",
    messages=[
        {"role": "system", "content": "You are a certified dermatologist. You will analyze the attached image of a person's face."},
        {
            "role": "user",
            "content": (
                "Analyze the skin in the image and describe skin tone, acne, dark circles, redness, blemishes, fine lines, "
                "and any other visible traits. Then, recommend a 3-step morning and night skincare routine using affordable "
                "products for this person. Include ingredients and lifestyle tips.\n\nRespond in the following format:\n"
                "Skin Features:\n[...] \nRecommendations:\n[...]"
            ),
            "image": image_base64,  
        }
    ]
)


    return response['message']['content']

if __name__ == "__main__":
    choice = input("Choose input method (1: Webcam, 2: Upload file): ")

    if choice == '1':
        img_path = capture_from_webcam()
    elif choice == '2':
        img_path = input("Enter the path of the image file: ")
        if not os.path.isfile(img_path):
            raise FileNotFoundError("Invalid file path")
    else:
        raise ValueError("Invalid choice")

    print("Analyzing skin...")
    result = analyze_skin(img_path)
    print("\n--- Skincare Recommendation ---\n")
    print(result)
