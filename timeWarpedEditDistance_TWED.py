class TWED():
    """
    Calculates the "Time-Warped Edit Distance" according to Marteau (2009), S. 312.
    
    @t1: Flat Numpy-array containing the data of the first time-series.
    @t2: Flat Numpy-array containing the data of the second time-series.
    @_lambda: Penalty for deletion.
    @_nu: Elasticity. _nu has to be >=0.
    
    Other variables:
    n: int, m: int -> length of t1, t2
    matrix: np.array with shape (n,m) -> Matrix for computation of the costs, whereby the low-right field matrix[n-1][m-1] contains the costs after computation.
    t1_time: np.array, t2_time: np.array -> Time-stamps of t1, t2. This will be np.arange(1, n+1, 1), np.arange(1, m+1, 1) per default.
    
    ____
    Use like:
    ```python
    
    TWED(np.array([10,20,30,40,50]), np.array([15,25,35,45,55])).calculateCosts()
    ```
    """
    
    __slots__ = ["t1", "t2", "_lambda", "_nu", "n", "m"]
    
    _LAMBDA = 0.001
    _NU = 0.5
    
    def __init__(self, t1: np.array, t2: np.array, _lambda: float = _LAMBDA, _nu: float = _NU) -> None:
        
        assert _nu >= 0, "Error: Set _nu >= 0!"
        
        self.t1      = t1
        self.t2      = t2
        self._lambda = _lambda
        self._nu     = _nu
        
        self.n       = len(t1)
        self.m       = len(t2)
        
        
    def _init_matrix(self, n: int, m: int) -> np.array:
        """
        Initialize matrix for operations.
        """
        matrix = np.zeros((n, m))
        
        for i in range(len(matrix)):
            matrix[i][0] = np.inf
            matrix[0][i] = np.inf
            
        matrix[0][0] = 0
        
        return matrix

    
    def calculateCosts(self) -> np.float:
        """
        Calculates the resulting costs according to TWED (hence, dissimilarity between t1 and t2).
        Returns the costs.
        """
            
        matrix  = self._init_matrix(self.n, self.m) #DP
        nu      = self._nu                    #Elasticity
        lam     = self._lambda                #Penalty for deletion
        t1_data = self.t1                     #Time-series data for t1
        t2_data = self.t2                     #Time-series data for t2
        t1_time = np.arange(1, self.n + 1, 1) #Time-stamps for t1
        t2_time = np.arange(1, self.m + 1, 1) #Time-stamps for t2
        _abs    = np.abs                      #Distance-measure (LP)
        
        
        for i in range(1, self.n):
            for j in range(1, self.m):
                #cost = _abs(t1_data[i] - t2_data[j]) #Irrelevant for computation, just added for completeness
                _deleteA = (
                            matrix[i-1][j] + 
                            _abs(t1_data[i-1] - t1_data[i]) +
                            nu*(t1_time[i] - t1_time[i-1]) + lam
                )
                _deleteB = (
                            matrix[i][j-1] + 
                            _abs(t2_data[j-1] - t2_data[j]) +
                            nu*(t2_time[j] - t2_time[j-1]) + lam
                )
                _match = (
                            matrix[i-1][j-1] + 
                            _abs(t1_data[i] - t2_data[j]) +
                            _abs(t1_data[i-1] - t2_data[j-1]) +
                            nu*(
                                _abs(t1_time[i] - t2_time[j]) + 
                                _abs(t1_time[i-1] - t2_time[j-1])
                            ) 
                )
                matrix[i][j] = min(_deleteA, _deleteB, _match)

        return matrix[self.n-1][self.m-1]    #-> Costs
