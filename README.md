# Lossy-Neuromorphic-Event-Compression
In this repository, the Python code for the Lossy Neuromorphic Event Compression is shared.

This code corresponds to the work done in the paper :- 
S. Banerjee, Z. W. Wang, H. H. Chopp, O. Cossairt and A. K. Katsaggelos, "Lossy Event Compression Based On Image-Derived Quad Trees And Poisson Disk Sampling," 2021 IEEE International Conference on Image Processing (ICIP), Anchorage, AK, USA, 2021, pp. 2154-2158, doi: 10.1109/ICIP42928.2021.9506546.

**Running the code**
1. Run **python main_starter.py** in order to run the PDS-LEC algorithm.
2. Provide path of the folder where your intensity files are located in **line #142** of main.py.
3. Provide path of the files where your event files are located in **line #175** of main.py.
4. **qt_events.py** is the file where the actual event compression code exists.
5. **Line #288** of main.py defines r_4 and other parameters.
6. For setting **Nbins_T** (temporal bins), change Nbins_T in **line #75** of qt_events.py. 


