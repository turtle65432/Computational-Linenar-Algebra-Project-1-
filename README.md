# Computational-Linenar-Algebra-Project-1-
This is the repository for our project.

Problem 1:

a) Apply k-means clustering with k = 2 to the training data. Then use the validation data
to assess the accuracy of your clustering. You will need to come up with a scheme to
determine the accuracy (i.e. a scheme to determine whether a patient in the validation
set has a malignant tumor or a benign tumor based on the clustering).

b) Embed the data in dimensions d ∈ {5, 10, 20} using Gaussian matrix embedding and
rerun k-means on the lower dimensional data set. What is the accuracy of the clustering
for each dimension d? What is the computational time averaged over 500 independent
runs?

c)  Embed the data in dimensions d ∈ {5, 10, 20} using the sparse random rrojection and
rerun k-means on the lower dimensional data set. What is the accuracy of the clustering
for each dimension d? What is the computational time averaged over 500 independent
runs?


Problem 2:

- Read in the data in train.txt into a matrix A whose rows correspond to the data
for each patient in the data set. The elements in a row correspond to the 30 features
measured for a patient.

- Read in the data in train values.txt into a vector b whose domain is the set of
patients and bi
is 1 if the specimen of patient i is malignant and it’s -1 if the specimen
is benign.

a) Use the QR algorithm to find the least-squares linear model for the data.

b) Apply the linear model from (a) to the data set validate.txt and predict the malignancy of the tissues. You will have to define a classifier function

                            C(y) = +1 if the prediction is non-negative
                                 = -1 otherwise

c) What is the percentage of samples that are incorrectly classified? Is it greater or
smaller than the success rate on the training data?

d) Embed the data in dimensions d ∈ {5, 10, 20} using Gaussian matrix embedding and
repeat the work in (a), (b) and (c) for each lower dimension d. What is the computational time averaged over 500 independent runs?

e) Embed the data in dimensions d ∈ {5, 10, 20} using sparse random projection and repeat the work in (a), (b) and (c) for each lower dimension d. What is the computational
time averaged over 500 independent runs?

Problem 3:

Apply k-means to the class music data songList.xlsx and use Class Roster to group
the class into 8 distinct music clusters.w