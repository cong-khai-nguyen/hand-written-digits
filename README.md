# Hand Written Digits
Link to test out: https://colab.research.google.com/drive/1TASX6lj9D2hK_m3jhNg5V9q15uVfIVo3?usp=sharing

# Description
Unlike others, this is the first project I implemented unsupervised ML algorithm. The objective of this project is to recognize hand-written digits and cluster them into groups of number from 0 to 9. Using Kmeans algorithm from scikit-learn library, I was able to
group the unlabeled datasets or written digits in this case into its apporirate cluster. 

# Install and Run the Project
This project requires installed Python library: scikit-learn

Note: the breast cancer data is included when we download scikit-learn. You can access it by including this import and function below:
```
from sklearn import datasets
cancer = datasets.load_digits()
