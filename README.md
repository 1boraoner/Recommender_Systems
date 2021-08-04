# Recommender_Systems

All the models below are based on the Dive Into the Deep Learning(d2l) book's Recommender Systems Chapter (Chapter 16)
One of the main goal of this Repository is to convert the code that is written in gluon into Pytorch. Other than that The models implemented are the state of the art recommendation system models.
The models that are investigated and implemented are Matrix Factorization Model, Factorization Machines, Deep Factorization Machines

0 - Loading the Data and Data preprocessing

  In all the models that are listed above are worked on the same dataset which is MovieLens100K.
  MovieLens100K dataset contins 943 unique Users and 1682 unique Movies with all the user-movie combination has a rating value. There are no NaN values.
  
  --Pytorch MovieLens Dataset is implemented. The dataset constructor has the option to convert the id values into OneHotEncoding and Rating values into Click Through    Problem where above 3 rating is 1 (clicked), 0 is not clicked.
  
  Besides the Dataset, there are 4 functions where makes the train-test split, creates the dataset and dataloader objects.


1- Matrix Factorization Model

  With Matrix Factorization model a rating problem is investigated. The main idea is that factoring the user-item ids into 2 seperate matrices where learned hidden features between the ids. The possible ratings can be guessed with the output and if the rating is above 3 the movie of a user can be recommended.
  Better to show with Image:
  ![resim](https://user-images.githubusercontent.com/43790905/128179323-cb7d8412-8686-4a45-9478-c0fd749d8106.png)

  
  Training config: 20 Epochs, Adam Optimizer,lr=0.002, Weights are initilised with Xavier Init.
  ![MF](https://user-images.githubusercontent.com/43790905/128178706-e7cd31c9-44d7-4321-8fd3-3b3c1a094774.jpg)
  ![trainMF](https://user-images.githubusercontent.com/43790905/128178779-4a569f2e-175d-4407-9d10-bb8be85c21ee.jpg)
  
  Finally, with this trained model any user and movie tuple's possible score can be found.
  
  
2- Factorization Machine

  Factorization Machines are models where used on top of the linear regression model and investigates the binary relation between the feature vectors.
  In the model, a latent matrix is constructed from a Factorization Layer. Then the vectors are dot prodcuted in two ways and being squared.
  With the formula:
  ![resim](https://user-images.githubusercontent.com/43790905/128179738-ec6e0486-901b-45c6-bd38-70e73a6f01b9.png)
  
  The <Vi, Vj> are the latent vectors of the two seperate features. With this the 2-way interactions of the features are learnt.
  
  Graphical Representation:
  ![resim](https://user-images.githubusercontent.com/43790905/128179896-5ee07ce4-97d9-4a23-80a0-f0deb2f21ab1.png)



