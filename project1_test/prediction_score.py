import numpy as np

submission1 = np.loadtxt('submission.csv', delimiter=',')
submission1 = submission1[:,1]

submission2 = np.loadtxt('submission4.csv', delimiter=',')
submission2 = submission2[:,1]
score_diff = np.sum((submission2-submission1)**2)/len(submission2)
print(score_diff)