# ConCodeRep
The implementation of "Adding Context to Source Code Representations for Deep 
Learning"

--------------------------------------------------------------------------------

## Dataset
This research is based on the dataset *SeSaMe*. 
<https://github.com/FAU-Inf2/sesame>

* If you want to replicate the experiments in this project, you have to download 
  the *SeSaMe* dataset and follow its authors' instructions to download the 11 
  projects. 

* You can run the `make pull` command directly in the directory ***sesame/src/***. 

  If you have a problem with this step, go to the original project link above to 

  find a solution.

## Jar
To do the next experiments, you must have *jar* packages for all the projects in 
the dataset. You can go one of the following two ways. 

* You can manually install them according to the documentation for each project 
or look for *jar* releases in a project's *github* repository.
* If you find the process above cumbersome, you can also use the *jar* packages 
  stored in the **jar_pack** folder.

## Java Callgraph
In this research, we use the tool *java-callgraph* to get the *callgraph* of 
source code. For more information, please visit this website. 
<https://github.com/gousiosg/java-callgraph>

We obtain the *callgraph* of each project by concatenating the *callgraphs* 
generated by the *jar* packages of the project. These *callgraphs* are stored in 
*cg* files, which can be viewed as common *txt* files.

 
