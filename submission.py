import pandas as pd
import utils

"""
Prepare the submission.csv file for submission to Kaggle for evaluation of model performance
"""

# Load preprocessing model, classifier and label encoder
print("Loading Models ...")
model_data = utils.loadModel("svd2000_svmOvR", "models")
# preprocess = model_data["model"]  # Valid only for triplet loss models and doc2cec
model = model_data["model"]
lb = model_data["labelencoder"]

# Load Test Data
print("Preparing Test Data ...")
test_id, test_ft = utils.loadPreprocessTest()

# #Alternatively load original test data and preprocess
# test_id, test_ft = utils.loadOfficialTestData()
# test_ft = [utils.preprocess(t) for t in test_ft]

# #Without preprocessing
# test_id, test_ft = utils.loadOfficialTestData()
# test_ft = [" ".join(t).lower() for t in test_ft]

# Predictions
print("Predicting ...")
test_out = model.predict(test_ft)
test_pred = lb.inverse_transform(test_out)

print ("Generate Submission File ... ")
sub = pd.DataFrame({'id': test_id, 'cuisine': test_pred}, columns=['id', 'cuisine'])
sub.to_csv('submission.csv', index=False)
