# Evaluating DöppleGANger

***here the keyword "the paper" refers to [this](https://pub.towardsai.net/generating-synthetic-sequential-data-using-gans-a1d67a7752ac) paper and "DöppleGANger" refers to [this](https://github.com/fjxmlzn/DoppelGANger) repository**

***I have used Python 3.6 and Tensorflow 1.4.0 for this project to make it compatible with Döppleganger.**

**The project contains a Trainer and a Generator file along side the gan folder from the DöppleGANger repository, "test" folder containing the generated training data and the dataset.**


On the Trainer side, first the data are being preprocessed and then fed to DöppleGANger to build and train, the preprocessing steps are as follows in detail:

- 20K rows of the dataset is being selected as opposed to the 100K rows in the paper to reduce the training time.
- I have set a range from 0-19 for the values of the 'Amount' column, otherwise there will be more than 18K categories in this column.
- I have measured the number of distinct categories for each categorical feature, the outcome is [20, 5, 44].
- Then I have a function that creates the NumpyArrays to carry the values of the features and attributes.
- The data_attribut array only contains the 'Balance' values.
- The data_all array contains the categorical and continuous columns.
- The processor function will oneHotEncode the 'Flag' and 'Description' columns and returns the sequences for each unique Account_id that contains the following columns:

`oneHotEncoded 'Flag'   oneHotEncoded 'Description'   'Amount'   'Balance'   'Dif'`

- Then I have put the values of each sequence in the right place in the Numpy arrays
- The Numpy arrays will get normalized using DöppleGANger's normalizer function.

Then the generation flags are being added to the features and the build will get initialized(Folders are being created to store the training outputs).
The discriminator functions of the DöppleGANger are being called
Then the model is being built and trained.

On the Generator side, we basically repeat the above steps, then we use the DöppleGANger's generator to generate the synthetic data based on the training data. then we denormalize and print out the results to compare with the originall data.

**Challenges:**

1: _Hazy processor:_
I have followed the paper to compose my code as recommended, and they have used a non open source Processor engine to pre-preprocess their data. I tried several times to contact Hazy corporation and asked them if I can get access to their libraries or if I can have a free student license but they never responded! Vera also suggested other open source synthetic data generation libraries such as the SDV of the MIT university, I digged into them but did not find them helpful to what I was trying to do, so I tried to keep on with my own approach of pre-processing the data, and it was quite challenging, I had no idea what the Hazy processor was doing with the data or what did it's outcome look like. Although I did know that it pre-process each sequence and reshape it in the right format.

2: _DöppleGANger:_
DÖppleGANger contains a util.py file where the normalization of the Numpy arrays take place, there it generates some additional attribute values and tries to concatenate them in the data_attribute array, since we only have one attribute 'Balance', and DöppleGANger always assumes at least 2 elements in both features and attribute output arrays, there will be a dimension mismatch, so you have to do some manipulations in the `def normalize_per_sample()` in the `util.py` file to make the code work, and if you do not do this correctly the whole `build()` process will face several issues in the other files such as the doppleganger.py and the network.py, this was the case for me and made me spend more than a week getting lost into the DöppleGANger codes looking for bugs... 

3: _After denormalizing the results:_
In the paprer, they have again used the Hazy processor after the denormalization to transform the data from the Numpy arrays back into sequences, and their code is quite strange since they are trying to concatenate the features array into the attributes array as follows:
`v = np.concatenate([np.zeros_like(attributes[i]), np.zeros_like(features[i])],axis=-1)`
and this is not possible since the attributes is supposed to either have dimension of (1000,) or (2,1000) whereas the dimensions of the features is supposed to be (1000,100.70). So at this point I decided to stop following the paper and do the rest on my own.

4: _Installation and version mismatch:_
What I needed for this work was Tensorflow 1.4.0 and a Python version lower than 3.7 .
I have an M1 macbook and an ASUS laptop runing windows10, at first I was trying everything on the macbook, but the new M1 machines come with lots of limitations regarding installation of Tensorflow, Numpy and etc. Hence having a local Python 3.6 and Tensorflow 1.4.0 was not possible, and also setting them up in a conda environment faced lots of pathing issues. At the end I managed to run Python 3.6 and Tensorflow 1.4.0 on a conda environmnet on my windows machine and run the code on jupyter notebook. 


**Results:**

My synthetic 'Balance' Values were not very simillar to the original values, but still acceptable. but the oneHotEncoded 'Flag' and 'Description' and the 'Amount' values came out quite strange. The inital values in the Numpy array where zeros and ones for those oneHotEncoded values, but the generated values are always floating points between zero and one, I tried to also oneHotEncode the 'Amount' values but the results were not much different.

**Conclusion:**

DöppleGANger is a great work, making snthetic data generation possible for areas where it is highly needed, although it comes with a number of issues that some of them where mentioned above, it would have been nice if they have made thir code compatible with newer versions of python and Tensorflow.
Their work is quite smart and mindblowing to some extent. using an MLP along side a recurrent network and decoupling the attribution generation and normalization, makes DöppleGANger unique in it's usage of GANs. Also it is very nice to have a normalizer and denormalizer embeded inside DöppleGANger. 
I learned very much working on this task and I appreciate the opportunity, unfortunately I wasted a lot of time trying to manipulate the DöppleGANger codes due to lack of explaination in the paper. I believe I will improve my results significantly with more time. I enjoyed working on the task and am quite eager to continue with the thesis, fix the results for the current task and then apply the code on different (medical) datasets and potentially come up with innovations to improve DöppleGANger.
