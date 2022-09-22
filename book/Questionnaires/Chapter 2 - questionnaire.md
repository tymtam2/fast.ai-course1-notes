1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
   Always-on cameras produce different style of images (saturation, contrast, etc.)
2. Where do text models currently have a major deficiency?
   Lack of logic and common sense
3. What are possible negative societal implications of text generation models?
   Too many to list all, or even most. One example is bots that are harder and harder to distinguish from humans.
4. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
   Staged releases starting with the model running in parallel to the current process.
5. What kind of tabular data is deep learning particularly good at?
   Recommendations
6. What's a key downside of directly using a deep learning model for recommendation systems?
   My view: Postitive feedback loops (popular things just get more popular)
   Book: They recommend things that you may have already watched. 
   *tt2: I don't fully understand this argument. I think the idea is that the recommendations systems show you what others like you like so they in fact can recommend things you don't even know you like.   
7. What are the steps of the Drivetrain Approach?
   1. Defined objective - What outcome am I trying to achieve?
   2. Levers - What inputs can I control?
   3. Data - What data we can control?
   4. Models - How the levers influence the objective?
8. How do the steps of the Drivetrain Approach map to a recommendation system?
   The *objective* of a recommendation engine is to drive additional sales by surprising and delighting the customer with recommendations of items they would not have purchased without the recommendation. The *lever* is the ranking of the recommendations. New *data* must be collected to generate recommendations that will cause new sales. This will require conducting many randomized experiments in order to collect data about a wide range of recommendations for a wide range of customers. This is a step that few organizations take; but without it, you don't have the information you need to actually optimize recommendations based on your true objective (more sales!).
   *tt2: I don't understand the mapping here*
9.  Create an image recognition model using data you curate, and deploy it on the web.
10. What is DataLoaders?
    Helper class to load data to train or fine-tune on
11. What four things do we need to tell fastai to create DataLoaders?
    * blocks=(ImageBlock, CategoryBlock), - What kind of data is it?
    * `get_items=get_image_files` - How to get the inputs?
    * `splitter=RandomSplitter(valid_pct=0.2, seed=42)` - How to divide the set into training and validation sets?
    * `get_y=parent_label` - How to get the lables?
12. What does the splitter parameter to DataBlock do?
    The splitter splits data into two or more sets, ususally training and validation sets. 
13. How do we ensure a random split always gives the same validation set?
    Provide the same seed. 
14. What letters are often used to signify the independent and dependent variables?
    `x` for inputs...aka independent variables and `y` for outputs aka dependent variables.  
15. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
    * *crop* takes the middle of the image preserving the aspect ratio
    * *pad* adds margin to preserve the aspect ratios 
    * *suish* ensures no cropping but may change the aspect ratio
16. What is data augmentation? Why is it needed?
    Augumentation is the process of modifying inputs when providing them for training. It helps preventing the model from learning to recognise the individual imager rather then recognising the features with the latter being what is desired. 
17. What is the difference between item_tfms and batch_tfms?
    `item_tfms` is applied on indfividual inputs on the CPU and `batch_tfms` is applied to batches and is exectued on the GPU.  
18. What is a confusion matrix?
    Confusion matrix is a matrix where one axis represents the actual lables and the other axis represents the predictions. 
19. What does export save?
    The export saves the model with its weigths.
20. What is it called when we use a model for getting predictions, instead of training?
    Inference 
21. What are IPython widgets?
    *(skipped)*
22. When might you want to use CPU for deployment? When might GPU be better?
    CPU will likely be the right approach - fast enough, easy to set up and cost effective. GPUs are more complex to use and the benefits are there only when parallelisation is taking place. 
    GPU might be better on high volume deployments.
23. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
    * Latency (but not always)
    * requirement for network connection
    * privacy
    * dealing with the server load:
      * various levels of load (cost of maintaining unused resources dring not-busy times), 
      * underutilisation of client resources
24. What are three examples of problems that could occur when rolling out a bear warning system in practice?
    > * Handling nighttime images, which may not appear in this dataset
    > * Ensuring results are returned fast enough to be useful in practice  
    > * Recognizing bears in positions that are rarely seen in photos that people post online
25. What is "out-of-domain data"?
    Data that is not present in the training set.
26. What is "domain shift"?
    Domain shift the effect where the data for which the model is used changes overt time. This means that the model's performance becomes worse and worse, unless the model is updated. 
27. What are the three steps in the deployment process?
    * Manual
      * Run model in parallel
      * Human check all predictions
    * Side-by-side
      * Careful human supervision
      * Time of geography limited
    * Gradual expansion
      * Good reporting system needed
      * Consider what could go wrong
