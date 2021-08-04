# Recommender_Systems

All the models below are based on the Dive Into the Deep Learning(d2l) book's Recommender Systems Chapter (Chapter 16)
One of the main goal of this Repository is to convert the code that is written in gluon into Pytorch. Other than that The models implemented are the state of the art recommendation system models.
The models that are investigated and implemented are Matrix Factorization Model, Factorization Machines, Deep Factorization Machines

0 - Loading the Data and Data preprocessing

  In all the models that are listed above are worked on the same dataset which is MovieLens100K.
  MovieLens100K dataset contins 943 unique Users and 1682 unique Movies with all the user-movie combination has a rating value. There are no NaN values.
  
  --Pytorch MovieLens Dataset is implemented. The dataset constructor has the option to convert the id values into OneHotEncoding and Rating values into Click Through    Rate Problem (CTR Prediction) where above 3 rating is 1 (clicked), 0 is not clicked.
  
  Besides the Dataset, there are 4 functions where makes the train-test split, creates the dataset and dataloader objects.
  
  CTR prediction is worked with Factorization Machine and DeepFM models. (d2l book does CTR prediction)

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
   
  Training Config: Adam Opt, lr:0.002 factorization 10
  ![FMson](https://user-images.githubusercontent.com/43790905/128200894-d2167228-595f-4599-8a99-1db83edf92d0.jpg)
  ![FMsonplot](https://user-images.githubusercontent.com/43790905/128200903-69fb5812-0de9-477e-8679-588bda0f1fe7.jpg)


3- DeepFactorization Machine

  DeepFactorization Machine is a combination of Deep Neural Network and Factorization Machine.
  One of the problem of the FM is that it only works with the binary interactions with DNN multiple interaction can be modelled
  
  Graphical Model:
  ![resim](https://user-images.githubusercontent.com/43790905/128202218-422c76a7-2b62-4ff1-8713-9a3bd9d8b9af.png)
  
  Training Config: Adam Opt, lr:0.002, factorization:10, DNN_dims:[50,40,25,10] drop_rate:0.1
  
  ![dfm1](https://user-images.githubusercontent.com/43790905/128201622-a840824c-791d-4eb2-bbe3-ac80e953dbb7.jpg)
  ![dfm11](https://user-images.githubusercontent.com/43790905/128201635-2b2644c6-4b82-4af8-a725-86b033395df1.jpg)



