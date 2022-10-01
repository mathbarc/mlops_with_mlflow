import mlflow
import dotenv
import cv2

if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    model = mlflow.pytorch.load_model("models:/rice_classifier/Production")
    model.eval()
    img = cv2.imread("datasets/Rice_Image_Dataset/Jasmine/Jasmine (1).jpg")
    result = model.predict(img)
    print(result)
