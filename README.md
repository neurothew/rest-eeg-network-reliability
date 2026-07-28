# rest-eeg-network-reliability
Codes to conduct reliability analysis to the network measures computed from resting-state EEG signals.

Read our two conference papers accepted in EMBC 2025 and EMBC 2026 respectively: 

Ma, M. K.-H., Fong, M. C.-M., & Wang, W. S. (2025). A Reliability Study in Resting-state EEG Network Characteristics: Frequency of Interest, Number of Oscillatory Cycles and Thresholding. 2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 1–7. https://doi.org/10.1109/EMBC58623.2025.11251614

Ma, M. K.-H. & Fong, M. C.-M. (accepted). A Path Toward Reproducibility: A Dual Perspective on Resting-State EEG Network Characteristics.

The codes are organized as follow:
```
code/
├── S1_main_preprocessing_intermediate.py             # Step 1
├── S2_main_filtering.py                              # Step 2
├── S3_main_epoching_and_computing_conn_mat.py        # Step 3
├── S4_computing_network_measures.py                  # Step 4
├── S5_combining_conn_dataframe.py                    # Step 5
└── S6_compute_reliability_essential.Rmd              # Step 6
```

Step 1 to 5 detail the procedures from preprocessing to obtaining a dataframe storing the all the network characteristics computed. After step 5, the dataframe will be loaded into step 6, which is a R markdown file for computing reliability.

Note that we made the present scripts available for transparency and readers' reference. They are not yet designed as a easy-to-run one-shot pipeline (but we plan to do so). Please take your time to adapt the path dependencies if you wish to replicate the computation process.