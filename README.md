# Neural_Network_Charity_Analysis

## Overview

With our knowledge of machine learning and neural networks, we used the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

    - EIN and NAME—Identification columns
    - APPLICATION_TYPE—Alphabet Soup application type
    - AFFILIATION—Affiliated sector of industry
    - CLASSIFICATION—Government organization classification
    - USE_CASE—Use case for funding
    - ORGANIZATION—Organization type
    - STATUS—Active status
    - INCOME_AMT—Income classification
    - SPECIAL_CONSIDERATIONS—Special consideration for application
    - ASK_AMT—Funding amount requested
    - IS_SUCCESSFUL—Was the money used effectively

###### Technologies Used:

- TensorFlow
- Keras and Keras-Tuner
- Scikit-learn
- Pandas
- Jupyter notebook
- Google Colaboratory
- MatplotLib and PyPlot

## Results

Detailed screenshots of every step in the flow are in the **Appendix** below.

#### Deliverable 1 - Preprocessing Data for a Neural Network Model

Using our knowledge of Pandas and the Scikit-Learn’s StandardScaler(), we preprocessed the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.

Steps:

1. Read in the charity_data.csv to a Pandas DataFrame. See Figure A1.
2. Drop the EIN and NAME columns. See Figure A2.
3. Determine the number of unique values for each column. See Figure A3.
4. Look at APPLICATION_TYPE value counts for binning. See Figure A4.
5. Visualize the value counts of APPLICATION_TYPE. See Figure A5.
6. Determine which values to replace if counts are less than ...? See Figure A6.
7. Look at CLASSIFICATION value counts for binning. See Figure A7.
8. Visualize the value counts of CLASSIFICATION. See Figure A8.
9. Determine which values to replace if counts are less than ..? See Figure A9.
10. Generate our categorical variable lists. See Figure A10.
11. Create a OneHotEncoder instance. See Figure A11.
12. Merge one-hot encoded features and drop the originals. See Figure A12.
13. Split our preprocessed data into features, target, train, and test arrays. See Figure A13.
14. Create a StandardScaler instances. See Figure A14.

#### Deliverable 2 - Compile, Train, and Evaluate the Model

Using our knowledge of TensorFlow, we designed a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Then we compiled, trained, and evaluated our binary classification model to calculate the model’s loss and accuracy.

Steps:

15. Define the model - deep neural net. See Figure A15.
16. Import checkpoint dependencies. See Figure A16.
17. Compile the model. See Figure A17.
18. Create a callback that saves the model's weights every epoch. See Figure A18.
19. Train the model. See Figure A19.
20. Evaluate the model using the test data. See Figure A20.
21. Export our model to HDF5 file. See Figure A21.

#### Deliverable 3 - Optimize the Model

Using our knowledge of TensorFlow, we attempted to optimize our model in order to achieve a target predictive accuracy higher than 75%.

## Summary

## Appendix

In order to avoid cluttering the main body of this report, all figures and code are presented in this Appendix. Some may be duplicated in the main body of the report to illustrate major points.

###### Deliverable 1

Figure A1 - Read the charity_data.csv into application_df

![read_csv.png](Images/read_csv.png)

Figure A2 - Drop the EIN and NAME columns.

![drop_ein_name_cols.png](Images/drop_ein_name_cols.png)

Figure A3 - Determine the number of unique values for each column.

![nunique.png](Images/nunique.png)

Figure A4 - Look at APPLICATION_TYPE value counts for binning.

![appl_types_value_counts.png](Images/appl_types_value_counts.png)

Figure A5 - Visualize the value counts of APPLICATION_TYPE.

![appl_types_density_plot.png](Images/appl_types_density_plot.png)

Figure A6 - Determine which values to replace if counts are less than ...?

![appl_types_binning.png](Images/appl_types_binning.png)

Figure A7 - Look at CLASSIFICATION value counts for binning.

![class_value_counts.png](Images/class_value_counts.png)

Figure A8 - Visualize the value counts of CLASSIFICATION.

![class_density_plot.png](Images/class_density_plot.png)

Figure A9 - Determine which values to replace if counts are less than ...?

![class_binning.png](Images/class_binning.png)

Figure A10 - Generate our categorical variable lists.

![categorical_vars.png](Images/categorical_vars.png)

Figure A11 -  Create a OneHotEncoder instance.

![one_hot_encoder.png](Images/one_hot_encoder.png)

Figure A12 - Merge one-hot encoded features and drop the originals.

![merged_df.png](Images/merged_df.png)

Figure A13 - Split our preprocessed data into features, target, train, and test arrays.

![split_data.png](Images/split_data.png)

Figure A14 - Create a StandardScaler instances.

![standard_scaler.png](Images/standard_scaler.png)

###### Deliverable 2

Figure A15 - Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.

![create_first_NN.png](Images/create_first_NN.png)

Figure A16 - Import checkpoint dependencies.

![import_checkpoint_deps.png](Images/import_checkpoint_deps.png)

Figure A17 - Compile the model.

![create_first_NN.png](Images/compile_first_NN.png)

Figure A18 - Create a callback that saves the model's weights every epoch.

![checkpoint_callback.png](Images/checkpoint_callback.png)

Figure A19 - Train the model.

![train_fit_firstNN.png](Images/train_fit_firstNN.png)

Figure A20 - Evaluate the model using the test data. See Figure A20.

![evaluate_firstNN.png](Images/evaluate_firstNN.png)

Figure A21 - Export our model to HDF5 file. See Figure A21.

![export_firstNN.png](Images/export_firstNN.png)

###### Deliverable 3

Figure A22 - reduce_count_vals()

![def_reduce_count_vals.png](Images/def_reduce_count_vals.png)

Figure A23 - def_do_one_hot()

![def_do_one_hot.png](Images/def_do_one_hot.png)

Figure A24 - def_do_scatter_plots()

![def_do_scatter_plots.png](Images/def_do_scatter_plots.png)

Figure A25 - def_do_scale()

![def_do_scale.png](Images/def_do_scale.png)

Figure A26 - def_build_model()

![def_build_model.png](Images/def_build_model.png)

Figure A27 - def_train_nn_model()

![def_train_nn_model.png](Images/def_train_nn_model.png)

Figure A28 - def_run_keras_tuner()

![def_run_keras_tuner.png](Images/def_run_keras_tuner.png)

Figure A29 - def_range_to_int()

![def_range_to_int.png](Images/def_range_to_int.png)

Figure A30 - def_chunk_ask()

![def_chunk_ask.png](Images/def_chunk_ask.png)

Figure A31 - def_encode_ask()

![def_encode_ask.png](Images/def_encode_ask.png)

Figure A32 - def_create_tuner_model()

![def_create_tuner_model.png](Images/def_create_tuner_model.png)

Figure A33 - run_keras_tuner2()

![def_run_keras_tuner2.png](Images/def_run_keras_tuner2.png)

Figure A34 - Optimization Run 1 Code

![opt_run1_code.png](Images/opt_run1_code.png)

Figure A35 - Optimization Run 1 Results

![opt_run1_results.png](Images/opt_run1_results.png)

Figure A36 - Optimization Run 2 Code

![opt_run2_code.png](Images/opt_run2_code.png)

Figure A37 - Optimization Run 2 Results

![opt_run2_results.png](Images/opt_run2_results.png)

Figure A38 - Optimization Run 3 (with Keras Tuner) Code and Results

![opt_run3_code_and_results.png](Images/opt_run3_code_and_results.png)

Figure A39 - Optimization Run 3 Best HyperParameters

![opt_run3_best_hypers.png](Images/opt_run3_best_hypers.png)

Figure A40 - Optimization Run 3 Best Models

![opt_run3_best_models.png](Images/opt_run3_best_models.png)
