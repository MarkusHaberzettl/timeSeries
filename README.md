# timeSeries
Scripts concerning time-series Data

## timeWarpedEditDistance_TWED.py
Contains a python implementation of Marteau's "Time-Warped Edit Distance" (TWED).

*P. Marteau, "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, no. 2, pp. 306-318, Feb. 2009.
doi: 10.1109/TPAMI.2008.76*

## memoryEfficient_TWED.py
Calculates TWED as well, but does not create the entire matrix for the calculation resulting in much less memory consumption. Additionally uses numba's jit for acceleration resulting in several hundred times more speed.
