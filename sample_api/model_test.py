import mlflow
import dotenv
import cv2
import torch
import numpy

if __name__=="__main__":
    dotenv.load_dotenv(".env")
    model = mlflow.pytorch.load_model("models:/rice_classifier/Staging")
    img = cv2.imread("/home/matheus/Projetos/tdc_mlops_with_mlflow/datasets/Rice_Image_Dataset/Arborio/Arborio (35).jpg")
    img = img.reshape(1,3,250,250).astype(numpy.float32)
    

    ten = torch.from_numpy(img)

    result = model(ten)
    print(result)