# Multimedia_Retrival
Multimedia retrieval consists of the theory, techniques, and tools that enable end users to search content of interest in large data collections consisting of (hyper)text, images, videos, audio, and 2D and 3D shapes.


create venv 
```
python -m venv .venv

.\.venv\Scripts\activate
```

Install libraries from requirments.txt 

```
pip install -r requirements.txt
```

obj_viewer.py: used to display a single mesh

show_one_class.py: used to display all meshes in a class

mesh_normalization.py: used to normalize all meshes in the database,

    input: step2_results/normalized
    output: step3_1_results/normalized

validate_norm.py : used to create the histogram to validate the success of the normalization.
    The results are group together in folder norm_analysis

f_e.py: used to extract the features, located in MR System folder.

    output: extracted features for each class are saved in one .csv file. All files all located in MR System/MR System/features

plot.py: used to plot histograms for distribution descriptors of a specific class

    input: a specific .csv file in MR System/MR System/features, user-defined
    output: histograms saved in MR System/plots

merge_all.py: used to merge all .csv files into one file and calculate the parameters for feature normalization

    input: all .csv files in MR System/MR System/features
    output: all_features.csv - features are normalized
            all_features_raw.csv - all values calculated by f_e.py and are not normalized
            normalization_params.csv - parameters used for feature normalization for querying mesh
            These files are located in MR System/database

step4.py: the logic of query implementation

query.py: Execute query

    input: a mesh, database/all_features.csv, database/normalization_params.csv
    output: similar mesh

fast_query.py, ANN.py: implement fast query
