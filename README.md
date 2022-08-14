# ConCodeRep
The implementation of "Adding Context to Source Code Representations for Deep 
Learning"

--------------------------------------------------------------------------------
The purpose of this project is to verify the impact of code call hierarchy on 
code representation and to explore its extent, so we do not focus on model 
innovation. *ASTNN*, one of the current SOTA models, is used as the basic 
experimental tool in this research.

## Dependencies
* python 3.6
* pandas 0.20.3
* gensim 3.6.0
* scikit-learn 0.19.1
* pytorch 1.5.0
* pycparser 2.18
* javalang 0.11.0

## Data
On the basis of *SeSaMe* dataset and related source code, two new datasets are 
obtained after preprocessing, which are used for clone detection and 
classification tasks respectively.

### Dataset
This research is based on the dataset *SeSaMe*. 
<https://github.com/FAU-Inf2/sesame>

* If you want to replicate the experiments in this project, you have to download 
  the *SeSaMe* dataset and follow its authors' instructions to download the 11 
  projects. 

* You can run the `make pull` command directly in the directory **sesame/src/**. 
  If you have a problem with this step, go to the original project link above to 
  find a solution.

### Jar
To do the next experiments, you must have *jar* packages for all the projects in 
the dataset. You can go one of the following two ways. 

* You can manually install them according to the documentation for each project 
or look for *jar* releases in a project's *github* repository.
* If you find the process above cumbersome, you can also use the *jar* packages 
  stored in the **jar_pack** folder.

### Java Callgraph
In this research, we use the tool *java-callgraph* to get the *callgraph* of 
source code. For more information, please visit this website. 
<https://github.com/gousiosg/java-callgraph>

We obtain the *callgraph* of each project by concatenating the *callgraphs* 
generated by the *jar* packages of the project. These *callgraphs* are stored in 
*cg* files, which can be viewed as common *txt* files.

### Context
We can find the context of each target code snippet through *callgraphs*. In the 
paper, we refer to the methods calling the target code snippet as *callers* and 
the methods called by the target code snippet as *callees*. (We used to call the 
*caller* and *callee* as *calling context* and *called context*. This 
designation is left over in the code.)

In the script **data_generation**, we can see the specific method of extracting 
the context code. The extracted results are stored in the 
**data/context_dataset.json** file, which is the most basic data we will use 
later.

### Preprocess
We need to process the dataset containing context into two kinds of datasets for 
clone detection and classification tasks. These two preprocessing are presented 
in two files, preprocess_clone and preprocess_class, respectively. They both 
parse code into abstract syntax trees, which are then converted into nested 
arrays via *word2vec*.

***The preprocessing here and the model below are adapted from ASTNN. If you 
want more details please go to the original project.***
<https://github.com/zhangj111/astnn>

## Model

The models used in this project are obtained by modifying *ASTNN*. Please refer to 
the paper for the specific modification details. The code of the training 
process is in the folder **src**, and the trained models are saved in the 
**models** folder.
