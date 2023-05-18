<h1 align="center">Automated Left Ventricular Ejection Fraction Assessment using Deep Learning</h1>
<h3 align="center">Revolutionizing Cardiac Analysis: Enhancing Efficiency and Accuracy through AI-driven Echocardiography</h3>


![Description of Image](https://github.com/OmdenaAI/topeka-chapter-ejection-fraction/blob/main/src/resources/LEVF-app.png)



<h2>Table of Contents</h2>

<ul>
  <li><a href="#background">Background</a></li>
  <li><a href="#problem">Problem</a></li>
  <li><a href="#project-goals">Project Goals</a></li>
  <li><a href="#learning-outcomes">Learning Outcomes</a></li>
  <li><a href="#project-structure">Project Structure</a></li>
</ul>


<h2 id="background">Background</h2>

Heart failure is a worldwide pandemic that affects at least 26 million people and is becoming more common. Heart failure continues to be a significant public health issue with rising expenses. A crucial indicator for the diagnosis and treatment of heart failure is the ejection fraction (EF). Ejection fraction measures how much blood leaves your heart with each contraction. It is only one of many tests that your doctor might use to find out how well your heart functions. The ejection fraction is often solely tested in the left ventricle. The heart's left ventricle serves as its primary pumping chamber. It forces oxygen-rich blood up into the aorta, the main artery in your body, to supply the rest of your body. 

Currently, the gold standard for determining left ventricular ejection fraction is cardiovascular magnetic resonance imaging (CMR). However, each cardiac MRI scan can cost anywhere between $100 and $5,000, or 5.5 times as much as an echocardiogram. Therefore, switching from CMR to echocardiograms to measure left ventricular ejection fraction would have significant health and financial benefits.

<h2 id="problem">Problem</h2>

Echocardiography is critical in cardiology. However, the full promise of echocardiography for precision medicine has been constrained by the requirement for human interpretation. In addition, the sonographer's experience is crucial for the human evaluation of heart function, and despite their years of training, there remains inter-observer variability. Echocardiograms have a complex multi-view format, which contributes to the fact that deep learning, a new method for image analysis, has not yet been widely used to analyse them.  An artificial intelligence (AI) solution can help identify cardiac structures with accuracy and automate LVEF measurement and myocardial motion with confidence.

<h2 id="project-goals">Project Goals</h2>
In this project, the Omdena Topeka Chapter  team aims to develop a Deep Learning model that will predict left ventricular ejection fraction (LVEF) values from ultrasound images . The project's primary goal is to accurately predict LVEF measurement.

With a duration of 8-weeks, this project aims to:

* Data Collection and Exploratory Data Analysis
* Preprocessing 
* Feature Extraction
* Model Development and Training
* Evaluate Model
* App development

<h2 id="learning-outcomes">Learning Outcomes</h2>

* Medical Image Processing
* Computer Vision
* Biomedical Image Analysis
* Project Managment
           

<h2 id="project-structure">Project Structure</h2>

    ├── LICENSE
    ├── README.md          <- The top-level README for developers/collaborators using this project.
    ├── original           <- Original Source Code of the challenge hosted by omdena. Can be used as a reference code for the current project goal.
    │ 
    │
    ├── reports            <- Folder containing the final reports/results of this project
    │   └── README.md      <- Details about final reports and analysis
    │ 
    │   
    ├── src                <- Source code folder for this project
        │
        ├── data           <- Datasets used and collected for this project
        │   
        ├── docs           <- Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
        │
        ├── references     <- Data dictionaries, manuals, and all other explanatory references used 
        │
        ├── resources     <- images, videos, and etc
        │
        ├── tasks          <- Master folder for all individual task folders
        │
        ├── visualizations <- Code and Visualization dashboards generated for the project
        │
        └── results        <- Folder to store Final analysis and modelling results and code.
--------

## Folder Overview

- Original          - Folder Containing old/completed Omdena challenge code.
- Reports           - Folder to store all Final Reports of this project
- Data              - Folder to Store all the data collected and used for this project 
- Docs              - Folder for Task documentations, Meeting Presentations and task Workflow Documents and Diagrams.
- References        - Folder to store any referneced code/research papers and other useful documents used for this project
- Resources         - Folder to images and videos used for this project
- Tasks             - Master folder for all tasks
  - All Task Folder names should follow specific naming convention
  - All Task folder names should be in chronologial order (from 1 to n)
  - All Task folders should have a README.md file with task Details and task goals along with an info table containing all code/notebook files with their links and information
  - Update the [task-table](./src/tasks/README.md#task-table) whenever a task is created and explain the purpose and goals of the task to others.
- Visualization     - Folder to store dashboards, analysis and visualization reports
- Results           - Folder to store final analysis modelling results for the project.


