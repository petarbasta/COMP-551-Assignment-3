from data_processing import DataProcessing
from linear_SVM import SVM
from cnn import CNN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():

    dp = DataProcessing()
    dp.hog()

    model = SVM()
    model.fit(dp.hog_features, dp.training_output_original)
    model.predict()



 #   X_train, X_test, y_train, y_test = train_test_split(dp.training_input, dp.training_output, test_size=0.33)
  #  model = CNN()
   # _, _, parameters = model.fit_model(X_train, y_train, X_test, y_test)







if __name__ == '__main__':
    main()