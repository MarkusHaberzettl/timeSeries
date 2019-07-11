from numba import jit
import numpy as np

class memTWED():
    """
    Calculates the "Time-Warped Edit Distance" following Marteau (2009), S. 312.
    Anyhow, for less memory usage only a matrix of shape (2, m) is created.
    
    @t1: Flat Numpy-array containing the data of the first time-series.
    @t2: Flat Numpy-array containing the data of the second time-series.
    @_lambda: Penalty for deletion.
    @_nu: Elasticity. _nu has to be >=0.
    
    Other variables:
    n: int, m: int -> length of t1, t2
    matrix: np.array with shape (2,m)
    
    ____
    Use like:
    ```python
    
    memTWED(np.array([10,20,30,40,50]), np.array([15,25,35,45,55])).calculateCosts()
    ```
    """
    
    __slots__ = ["t1", "t2", "_lambda", "_nu", "n", "m"]
    
    _LAMBDA = 0.001
    _NU = 0.5
    
    def __init__(self, t1: np.array, t2: np.array, _lambda: float = _LAMBDA, _nu: float = _NU, failOnDifferingLengths: bool = False) -> None:
        
        assert _nu >= 0, "Error: Set _nu >= 0!"
        
        self.t1      = t1
        self.t2      = t2
        self._lambda = _lambda
        self._nu     = _nu
        
        self.n       = len(t1)
        self.m       = len(t2)
        
        if failOnDifferingLengths:
            assert self.n == self.m, f"Error, n != m!: {self.n}, {self.m}"
        elif self.n < self.m:
            print("Warning: n < m will result in long runtimes!")
        
        
    def _init_matrix(self) -> np.array:
        """
        Initialize matrix for operations.
        """
        matrix = np.zeros((2, self.n))
        
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        
        return matrix

    
    def calculateCosts(self):
        """
        Calculates the resulting costs according to TWED (hence, dissimilarity between t1 and t2).
        Returns the costs.
            
        matrix  = self._init_matrix()         #DP
        nu      = self._nu                    #Elasticity
        n       = self.n                      #Number of timesteps in t1
        m       = self.m                      #Number of timesteps in t2
        lam     = self._lambda                #Penalty for deletion
        t1_data = self.t1                     #Time-series data for t1
        t2_data = self.t2                     #Time-series data for t2
        """
        return self._calculateCosts(self._init_matrix(), self.n, self.m, self._nu, self._lambda, self.t1, self.t2)
    
    
    @staticmethod
    @jit
    def _calculateCosts(matrix, n, m, nu, lam, t1_data, t2_data) -> np.float:
        """
        Calculates the resulting costs according to TWED (hence, dissimilarity between t1 and t2).
        Returns the costs.
            
        matrix  = self._init_matrix()         #DP
        nu      = self._nu                    #Elasticity
        lam     = self._lambda                #Penalty for deletion
        t1_data = self.t1                     #Time-series data for t1
        t2_data = self.t2                     #Time-series data for t2
        """
        _abs    = np.abs                      #Distance-measure (LP)
        
        
        for i in range(1, n):
            mi = i % 2 #Current Row
            ai = abs(mi-1) #Previous Row
            
            for j in range(1, m):
                
                _deleteA = (
                            matrix[ai, j] + 
                            _abs(t1_data[i-1] - t1_data[i]) +
                            nu*(i - (i-1)) + lam
                )
                _deleteB = (
                            matrix[mi, j-1] + 
                            _abs(t2_data[j-1] - t2_data[j]) +
                            nu*(j - (j-1)) + lam
                )
                _match = (
                            matrix[ai, j-1] + 
                            _abs(t1_data[i] - t2_data[j]) +
                            _abs(t1_data[i-1] - t2_data[j-1]) +
                            nu*(
                                _abs(i - j) + 
                                _abs((i-1) - (j-1))
                            ) 
                )
                matrix[mi, j] = min(_deleteA, _deleteB, _match)
            else:
                matrix[0, 0] = np.inf

        return matrix[mi, m-1]    #-> Costs
