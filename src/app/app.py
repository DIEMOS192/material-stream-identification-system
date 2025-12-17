import cv2
import joblib as jl


def main():
    model_scaler = jl.load('../saved_models/svm_scaler.joblib')
    model_normalizer = jl.load('../saved_models/svm_normalizer.joblib')
    better_model = jl.load('../saved_models/svm_model.joblib')

    

    pass




if __name__ == "__main__":
    main()








