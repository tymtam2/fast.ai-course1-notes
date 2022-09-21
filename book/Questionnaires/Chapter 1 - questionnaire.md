1. Do you need these for deep learning?

   - Lots of math T / F 
     *A: False*
   - Lots of data T / F 
     *A: False*
   - Lots of expensive computers T / F
     *A: False*
   - A PhD T / F
     *A: False*

2. Name five areas where deep learning is now the best in the world.
   1. Natural langue processing, e.g.: 
      1.  Speech recognition (btw. OpenAI just releasing (Whisper)[https://openai.com/blog/whisper/])
      2.  Translation 
   2. Computer vision: 
   3. Games: 
      1. AlphaStar - starcraft
      2. AlphaZero - chess, go
   4. Computational biology: 
      1. [Protein-folding](https://www.deepmind.com/research/highlighted-research/alphafold) 
   5. Recommendation systems: de facto standard
3. What was the name of the first device that was based on the principle of the artificial neuron?
   Mark 1 perceptron
4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?
   Parallel Distributed Processing (PDP) by David Rumelhart, James McClellan, and the PDP Research Group, released in 1986 by MIT Press.
   The book defined parallel distributed processing as requiring:
    1. A set of processing units
    2. A state of activation
    3. An output function for each unit
    4. A pattern of connectivity among units
    5. A propagation rule for propagating patterns of activities through the network of connectivities
    6. An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
    7. A learning rule whereby patterns of connectivity are modified by experience
    8. An environment within which the system must operate
    (Retention of this must be hovering just above 0% of students :))
5. What were the two theoretical misunderstandings that held back the field of neural networks?
   1. >. In the same book, they also showed that using multiple layers of the devices would allow these limitations to be addressed. Unfortunately, only the first of these insights was widely recognized. As a result, the global academic community nearly entirely gave up on neural networks for the next two decades.
   2. >However, again a misunderstanding of the theoretical issues held back the field. In theory, adding just one extra layer of neurons was enough to allow any mathematical function to be approximated with these neural networks, but in practice such networks were often too big and too slow to be useful.
6. What is a GPU?
   A graphics processing unit, aka graphics card. Very good at doing things in parallel.
7. Open a notebook and execute a cell containing: `1+1`. What happens?
   In a code cell it will print 2 below the cell after running it in a remote session (remote as in: not in the browser).
8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.
9.  Complete the Jupyter Notebook online appendix.
    *Unable to find the appendix*
10. Why is it hard to use a traditional computer program to recognize images in a photo?
    Traditional computer programs are sets of rules for the computer to follow but the rules of recognition is not understood. 
11. What did Samuel mean by "weight assignment"?
    > First, we need to understand what Samuel means by a weight assignment.
    > Weights are just variables, and a weight assignment is a particular choice of values for those variables. The program's inputs are values that it processes in order to produce its results—for instance, taking image pixels as inputs, and returning the classification "dog" as a result. The program's weight assignments are other values that define how the program will operate.
12. What term do we normally use in deep learning for what Samuel called "weights"?
    Parameters
13. Draw a picture that summarizes Samuel's view of a machine learning model.
    ```
    Inputs -> Model -> Result
       ^------Update----╯  
    ```
14. Why is it hard to understand why a deep learning model makes a particular prediction?
    The function the model represents is almost always non-trivial

15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
    From [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)):
    > Universal approximation theorems imply that neural networks can represent a wide variety of interesting functions when given appropriate weights. On the other hand, they typically do not provide a construction for the weights, but merely state that such a construction is possible.  
16. What do you need in order to train a model?
    1.  Data
        1.  Data itself
        2.  Labels (often)
    2.  Architecture 
        1.  Structure
        2.  Way to backpropagate
    3.  Loss function    
    4.  A Jupyter notebook ;)
17. How could a feedback loop impact the rollout of a predictive policing model?
    > Another critical insight comes from considering how a model interacts with its environment. This can create feedback loops, as described here:

    > A predictive policing model is created based on where arrests have been made in the past. In practice, this is not actually predicting crime, but rather predicting arrests, and is therefore partially simply reflecting biases in existing policing processes.
    Law enforcement officers then might use that model to decide where to focus their police activity, resulting in increased arrests in those areas.
    Data on these additional arrests would then be fed back in to retrain future versions of the model.

    > This is a positive feedback loop, where the more the model is used, the more biased the data becomes, making the model even more biased, and so forth.

    > Feedback loops can also create problems in commercial settings. For instance, a video recommendation system might be biased toward recommending content consumed by the biggest watchers of video (e.g., conspiracy theorists and extremists tend to watch more online video content than the average), resulting in those users increasing their video consumption, resulting in more of those kinds of videos being recommended. 
18. Do we always have to use 224×224-pixel images with the cat recognition model?
    No
19. What is the difference between classification and regression?
    Classification has a distinct number of possible outputs; and 
    Regression produces an output from a continuous range (eg. a number)
20. What is a validation set? What is a test set? Why do we need them?
    Validation and test sets are used to test how the model works on inputs it didn't see.
    Validation model is used in the train,check,improve development loop
    Test model is reserved for testing the model after improvements are made. 
    They are two distinct sets to prevent the situation that the model is optimised only for the validation set and in practice it would not perform well in real life scenarios.
21. What will fastai do if you don't provide a validation set?
    Set asside 20% automatically
22. Can we always use a random sample for a validation set? Why or why not?
    We can, but it may not produce the best results. The validation set should be sufficiently different from the traninig set so that it can be considered to be not seen by the model. In an exmaple where the model is tasked with recognising if the subject holds a mobile phone the training set should not contain pictures of subjects from the validation set because the model could (and probably would) learn to recognise the subject and not the holding of the mobile phone. 
23. What is overfitting? Provide an example.
    Overfitting is the phonomenon where the neural net learns to recognise the data (e.g. a particular image) instead of the pattern (e.g. a cat). An example of overfitting could be a model that learns to recognise particular photos rather than the features on the set. In other words the model learns the ansers for a particular photo. 
24. What is a metric? How does it differ from "loss"?
    Metric is measure of how well model performs. Loss is used for the model to update its weights. It may be ok for metric and loss to use the same function. 
25. How can pretrained models help?
    It can help significantly.
    Using a pretrained model means that we 1) can use a state-of-the-art architecture, 2) don't need to have tons of data to train the model 3) do not require the time and processing power to train the model from scratch  
26. What is the "head" of a model?
    The head of the model is the part of the neural net which is retrained in the process of fine-tuning. 
27. What kinds of features do the early layers of a CNN find? How about the later layers?
    The early layers find lower level feautres like edges, color boundraries. The later layers find features that are build from the lower level features. 
28. Are image models only useful for photos?
    Yes and no. They are useful only for images, but many types of inputs can be transformed into images. 
29. What is an "architecture"?
    It the way the model is build. It's all the things that make model what it is. For a simple neural net it could be number of layers, number of neurons (per layer) and how the neurons are connected.
30. What is segmentation?
    Segmenation is clasffifying parts of the image (e.g. pixels) into categories.
31. What is `y_range` used for? When do we need it?
    `y_range` is the parmeter used to constrain the range of the output of the model. The most common example probably is the rating systems where the range is restricted to numbers from 1 to 5 (there is a neuance here which will be covered in the later part of the book).
32. What are "hyperparameters"?
    Hyperparameters are the paramaters that describe the neural network and/or the training process, not the prameters used for producting the output. Examples of heyperparameters are: 
    1. Number of layers
    2. Number of epochs or learning rates 
33. What's the best way to avoid failures when using AI in an organization?
    The best way to avoid failures is use a set of measures and the best of these measures is using a good test set. 