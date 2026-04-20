import numpy as np
from typing import Self
from numpy.typing import NDArray,ArrayLike

class AngularCorrelator:
    r'''
    Class to compute angular cross-correlation from data and masks given on a uniform polar grid.

    Attributes:
        n_radial_samples (int): Number of radial sampling points.
        n_angular_samples (int): Number of uniform angular sampling points.
        use_cuda (bool): Whether or not to use cuda for fft computations.

    Examples:
        ```py
        import numpy as np
        from extra.utils.xcca import AngularCorrelator

        n_q,n_phi = 32,64
        data = np.random.rand(n_q,n_phi)
        mask = np.random.rand(n_q,n_phi)>0.7
    
        a = AngularCorrelator(n_q,n_phi)
    
        # Compute mask corrected angular cross-correlation function.
        ccf, ccf_mask = a.ccf(data,mask = mask)
    
        # Compute first 11 Fourier coefficients of mask corrected angular cross-correlation function.
        ccn, ccn_mask = a.ccn(data,mask = mask,max_order=11)
        ```
    '''    
    def __init__(self,n_radial_samples:int=256,n_angular_samples:int=1024,use_cuda:bool=False):
        self.n_radial_samples = n_radial_samples
        self.n_angular_samples = n_angular_samples
        self.use_cuda = use_cuda
        
        if use_cuda:
            import cupyx.scipy.fft as cufft
            self.rfft = cufft.rfft
            self.irfft = cufft.irfft
        else:
            self.rfft = np.fft.rfft
            self.irfft = np.fft.irfft
            
        self.ccf_workspace = np.empty((3,self.n_radial_samples,self.n_angular_samples),dtype = float)
        self.ccn_workspace = np.empty((3,self.n_radial_samples,self.n_angular_samples//2+1),dtype = complex)
        self.mask_workspace = np.empty((self.n_radial_samples,self.n_angular_samples),dtype = bool)

    @staticmethod 
    def ccf_mask_correction(ccf_data:NDArray[np.float64], ccf_mask:NDArray[np.bool]) -> tuple((NDArray[np.float64],NDArray[np.bool])):
        r"""Apply mask correction to cross-correlation

        Applies mask correction to ccf computed from data*mask (ccf_data) using ccf computed from only the mask (ccf_mask).
        Correction is done according to: J Appl Crystallogr, 2024, 57, 324 (Equation 16)
        https://journals.iucr.org/j/issues/2024/02/00/yr5118/yr5118.pdf

        Mask convention: masked values are 0 while unmasked values are 1.

        Args:
            ccf_data: Cross-corelation computed from data*mask.
            ccf_mask: Cross-correlation computed form mask.

        Returns:
            (Corrected cross-correlation, Mask of corrected cross-correlation).
        """
        ccf_data=ccf_data.real
        ccf_mask=ccf_mask.real

        # ccf_mask shoud only contain multiples of 1/n_phis as values
        # make sure there are no values lower than 1/n_phis.
        # Use 1/(2*n_phis) as threshold instead of 1/n_phi to be insensitve to rounding errors.

        n_phis = ccf_mask.shape[-1]
        nonzero_mask = (ccf_mask>=1/(2*n_phis))
        np.divide(ccf_data, ccf_mask, out=ccf_data, where=nonzero_mask)
        return ccf_data,nonzero_mask

    def ccn_from_ccf(self,ccf:NDArray[np.float64],max_order: int|None = None) -> NDArray[np.float64]:
        r"""Compute Fourier series coefficients of cross-correlation.

        Args:
            ccf: (n_q,n_q,n_phi): Cross-correlation function $C(q_1,q_2,\phi)$.
            max_order: Maximum computed Fouerier series order. Defaults to None.

        Returns:
            NDArray[np.float64]: (n_q,n_q,n_orders): Fourier coefficients $C_n(q_1,q_2)$.
        """
        rfft = self.rfft
        if max_order is None:
            bw = ccf.shape[-1]//2 + 1
            ccn_workspace = self.ccn_workspace[0,:,:bw]
        else:
            bw = max_order+1
            if  ccf.shape[-1] == 2*max_order:                
                ccn_workspace = self.ccn_workspace[0,:,:bw]
            else:
                ccn_workspace = self.ccn_workspace[0]
            
        ccn = np.empty(ccf.shape[:2]+(bw,),dtype = complex)
        for q1 in range(self.n_radial_samples):
            rfft(ccf[q1,q1:],axis = -1,norm='forward',out=ccn_workspace[q1:])
            ccn[q1,q1:] = ccn_workspace[q1:,:bw]
            # Fill rest by symmetry
            ccn[q1:,q1] = ccn[q1,q1:].conj()
        return ccn
    
    def ccf_from_ccn(self,ccn:NDArray[np.complex128],max_order: int|None = None) -> NDArray[np.float64]:
        r"""Compute cross-correlation function from its Fourier coefficients.

        Args:
            ccn: (n_q,n_q,n_orders): Fourier coefficients $C_n(q_1,q_2)$.
            max_order: Maximum considered Fourier coefficient order. Defaults to None.

        Returns:
            (n_q,n_q,n_phi): Cross-correlation function $C(q_1,q_2,\phi)$.
        """
        irfft = self.irfft
        if max_order is None:
            N = (ccn.shape[-1]-1)*2
        else:
            ccn = ccn[...,:max_order+1]
            N = max_order*2
            
        ccf = np.empty(ccn.shape[:2]+(N,),dtype = complex)
        for q1 in range(self.n_radial_samples):
            irfft(ccn[q1,q1:],n=N,axis = -1,norm='forward',out=ccf[q1,q1:])
            # Fill rest by symmetry
            ccf[q1:,q1,0] = ccf[q1,q1:,0]
            ccf[q1:,q1,1:] = ccf[q1,q1:,-1:0:-1]
        return ccf
    
    def _compute_ccf(self, data:NDArray[np.float64])->NDArray[np.float64]:
        r"""Compute cross-correlation function from polar image.

        Widh data = $I(q,\phi)$ this function computes
        $C(q_1,q_2,\phi)=\mathfrac{F}\left(I_n(q_1) I_n(q_2)^*\right)$.

        Using the symmetry $C(q1,q2,phi) = C(q2,q1,-phi)$

        Args:
            data: (n_q,n_phi): image data on a uniform polar grid.
        Returns:
            (n_q,n_q,n_phi): Cross-correlation function $C(q_1,q_2,\phi)$.
        """
        n_q = self.n_radial_samples
        
        fn = self.rfft(data,axis=-1,norm='forward')
        fn_conj = fn.conj()

        ccf = np.zeros((n_q,n_q,self.n_angular_samples),dtype=float)

        ircht = self.irfft
        mult = np.multiply
        N = self.n_angular_samples
        ccn_workspace = self.ccn_workspace[0]
        for q1 in range(n_q):
            mult(fn[q1],fn_conj[q1:],out = ccn_workspace[q1:])
            ircht(ccn_workspace[q1:],n=N,axis=-1,norm='forward',out = ccf[q1,q1:])
            # Fill symmetric part. 
            ccf[q1:,q1,0] = ccf[q1,q1:,0]
            ccf[q1:,q1,1:] = ccf[q1,q1:,-1:0:-1]
        return  ccf

    def _compute_ccn_masked(self,data:NDArray[np.float64],mask:NDArray[np.bool],max_order:int|None = None) -> tuple[NDArray[np.complex128],NDArray[np.bool]]:
        r"""Compute mask corrected Fourier cofficients of the cross-correlation function.

        Args:
            data: (n_q,n_phi): Image data on uniform polar grid.
            mask: Image mask on uniform polar grid.
            max_order: Maximum computed Fourier series coefficient. Defaults to None.

        Returns:
            ((n_q,n_q,max_order+1) , (n_q,n_q)): (Fourier coefficients $C_n(q_1,q_2)$, Mask of Fourier coefficients.).
        """
        n_q = self.n_radial_samples
        
        # Compute harmonic coefficients of image and mask
        fmask = mask.astype(float)
        fn = self.rfft(data*fmask,axis = -1, norm='forward')
        mn = self.rfft(fmask,axis = -1, norm='forward')

        fn_conj = fn.conjugate()
        mn_conj = mn.conjugate()

        if max_order is None:
            bw = self.n_angular_samples//2+1
        else:
            bw = max_order+1

        ccn_workspace = self.ccn_workspace
        ccf_workspace = self.ccf_workspace
        mask_workspace = self.mask_workspace
        ccn = np.zeros((n_q,n_q,bw),dtype=complex)
        ccn_mask = np.zeros((n_q,n_q,bw),dtype=bool)

        #map numpy methods
        mult = np.multiply
        divide = np.divide
        irfft = self.irfft
        rfft = self.rfft
        N = self.n_angular_samples
        
        # start loop over q1 of C(q1,q2,phi)
        for q1 in range(n_q):
            # Compute parts of the cross correlation of image and mask (only unsymmetric part)
            mult(fn[q1],fn_conj[q1:],out = ccn_workspace[0,q1:])
            mult(mn[q1],mn_conj[q1:],out = ccn_workspace[1,q1:])
            irfft(ccn_workspace[0,q1:],n=N,axis=-1,norm='forward',out = ccf_workspace[0,q1:])
            irfft(ccn_workspace[1,q1:],n=N,axis=-1,norm='forward',out = ccf_workspace[1,q1:])
            
            # compute the boolean mask at wich ccf is defined (i.e. could be computed)
            mask_workspace[q1:]=ccf_workspace[1,q1:]>1/(2*N)
            # correct the computed image cross correlation by dividing out the mask correlation
            divide(ccf_workspace[0,q1:],ccf_workspace[1,q1:],where = mask_workspace[q1:],out=ccf_workspace[2,q1:])
            
            ccn_mask[q1,q1:] = np.all(mask_workspace[q1:],axis = -1)[...,None]
            rfft(ccf_workspace[2,q1:],axis = -1, norm='forward',out = ccn_workspace[2,q1:])
            ccn[q1,q1:]=ccn_workspace[2,q1:,:bw]
            
            # Use the Symmetrie C(q1,q2,phi)=C(q2,q1,-phi) which imposes
            # the symmetry Cn(q1,q2) = Cn(q2,q1)^* on its harmonic coefficents
            ccn[q1+1:,q1,:] = ccn[q1,q1+1:,:].conj()
            ccn_mask[q1+1:,q1] = ccn_mask[q1,q1+1:]
            
        return ccn, ccn_mask
    
    def _compute_ccf_masked(self,data:NDArray[np.float64],mask:NDArray[np.bool],max_order:int|None = None) -> tuple[NDArray[np.float64],NDArray[np.bool]]:
        r"""Compute the mask corrected cross-correlation function.

        Args:
            data: Image data on uniform polar grid.
            mask: (n_1,n_phi): Image mask on uniform polar grid.
            max_order: Maximum considered Fourier series order. Defaults to None.

        Returns:
            ((n_q,n_q,2*max_order), (n_q,n_q,2*max_order)): (Cross-correlation function $C(q_1,q_2,\phi)$, Mask of cross-correlation function.)
                If max_order == None the last dimension has size n_phi.
        """
        
        r"""Compute the mask corrected cross-correlation function.

        Parameters
        ----------
        data : NDArray[np.float64]
            Image data on uniform polar grid.
        mask : NDArray[np.bool]
            (n_1,n_phi): Image mask on uniform polar grid.
        max_order : int|None
            Maximum considered Fourier series order.

        Returns
        -------
        tuple(NDArray[np.float64],NDArray[np.bool])
            ((n_q,n_q,2*max_order), (n_q,n_q,2*max_order)): (Cross-correlation function $C(q_1,q_2,\phi)$, Mask of cross-correlation function.)
            If max_order == None the last dimension has size n_phi.
        
        """
        if max_order is None:
            ccf,ccf_mask = self._compute_ccf_masked_full(data,mask)
        else:
            ccn,ccn_mask = self._compute_ccn_masked(data,mask,max_order=max_order)
            ccf = self.ccf_from_ccn(ccn,max_order=max_order)
            ccf_mask = ccn_mask
        return ccf,ccf_mask
    def _compute_ccf_masked_full(self,data:NDArray[np.float64],mask:NDArray[np.bool]) -> tuple[NDArray[np.float64],NDArray[np.bool]]:
        r"""Compute Cross-correlation function using all harmonic orders.

        Args:
            data: (n_q,n_phi): Image data on uniform polar grid.
            mask: (n_q,n_phi): Image mask on uniform polar grid.

        Returns:
            (n_q,n_q,n_phi),(n_q,n_q,n_phi): (Cross-correlation $C(q_1,q_2,\phi)$, Mask of cross-correlation).
        """
        # Compute harmonic coefficients of image and mask
        fmask = mask.astype(float)
        fn = self.rfft(data*fmask,axis = -1, norm='forward')
        mn = self.rfft(fmask,axis = -1, norm='forward')

        fn_conj = fn.conjugate()
        mn_conj = mn.conjugate()
        
        bw = self.n_angular_samples//2+1
        N = self.n_angular_samples
        n_q = self.n_radial_samples
        
        ccn_workspace = self.ccn_workspace
        ccf_workspace = self.ccf_workspace
        mask_workspace = self.mask_workspace
        ccf = np.zeros((n_q,n_q,N),dtype=complex)
        ccf_mask = np.zeros((n_q,n_q,N),dtype=bool)
        
        #map numpy methods
        mult = np.multiply
        divide = np.divide
        irfft = self.irfft
        rfft = self.rfft
        
        # start loop over q1 of C(q1,q2,phi)
        for q1 in range(n_q):
            # Compute parts of the cross correlation of image and mask (only unsymmetric part)
            mult(fn[q1],fn_conj[q1:],out = ccn_workspace[0,q1:])
            mult(mn[q1],mn_conj[q1:],out = ccn_workspace[1,q1:])
            irfft(ccn_workspace[0,q1:],n=N,axis=-1,norm='forward',out = ccf_workspace[0,q1:])
            irfft(ccn_workspace[1,q1:],n=N,axis=-1,norm='forward',out = ccf_workspace[1,q1:])
            
            # compute the boolean mask at wich ccf is defined (i.e. could be computed)
            ccf_mask[q1,q1:]=ccf_workspace[1,q1:]>1/(2*N)
            # correct the computed image cross correlation by dividing out the mask correlation
            divide(ccf_workspace[0,q1:],ccf_workspace[1,q1:],where = ccf_mask[q1,q1:],out=ccf[q1,q1:])
            
            # Fill rest using the Symmetrie C(q1,q2,phi)=C(q2,q1,-phi)
            ccf[q1:,q1,0] = ccf[q1,q1:,0]
            ccf[q1:,q1,1:] = ccf[q1,q1:,-1:0:-1]
            ccf_mask[q1:,q1,0] = ccf_mask[q1,q1:,0]
            ccf_mask[q1:,q1,1:] = ccf_mask[q1,q1:,-1:0:-1]                                
        return ccf,ccf_mask
    
    def ccn(self,data:NDArray[np.float64], mask: NDArray[np.bool]|None  = None, max_order:int|None = None) -> NDArray[np.complex128]|tuple[NDArray[np.complex128],NDArray[np.bool]]:
        r"""Compute Fourier coefficients of the corss-correlation funcion.

        Lowering max_order does not save computation time but simply cuts the output to the required maximum order therefore saving RAM.

        Args:
            data: (n_q,n_phi): Image data on uniform polar grid.
            mask: If the (n_q,n_phi) Image mask array is provided this
                routine automatically applies mask correction to the
                computed Fourier coefficients.. Defaults to None.
            max_order: Maximum computed Fourier coefficient order. Defaults to None.

        Returns:
            If mask was provided it returns both the mask corrected
                Fourier coefficients and their mask. Other wise it just
                returns the Fourier coefficients.

        Examples:
            ```py
            import numpy as np
            from extra.applications.xcca import AngularCorrelator
            n_q = 128
            n_phi = 256
            a = AngularCorrelator(n_q,n_phi)

            data = np.random.rand(n_q,n_phi)
            mask = np.random.rand(n_q,n_phi)>0.7
            ccn,ccn_mask = a.ccn(data,mask,max_order=31)
            ```
        """
        if mask is None:
            ccf = self._compute_ccf(data)
            ccn = self.ccn_from_ccf(ccf,max_order = max_order)
            return ccn
        else:
            ccn,ccn_mask = self._compute_ccn_masked(data,mask,max_order = max_order)
            return ccn,ccn_mask
    
    def ccf(self,data:NDArray[np.float64], mask:NDArray[np.bool]|None = None, max_order:int|None = None) -> NDArray[np.float64]|tuple[NDArray[np.float64],NDArray[np.bool]]:
        r"""Compute the corss-correlation funcion.

        Lowering max_order does not save computation time but simply cuts the output to the required maximum order therefore saving RAM.

        Args:
            data: (n_q,n_phi): Image data on uniform polar grid.
            mask: If the (n_q,n_phi) Image mask array is provided this
                routine automatically applies mask correction to the
                computed Fourier coefficients.. Defaults to None.
            max_order: Maximum computed Fourier coefficient order. Defaults to None.

        Returns:
            If mask was provided it returns both the mask corrected
                cross-correlation and its mask. Other wise it just
                returns the cross-correlation.

        Examples:
            ```py
            import numpy as np
            from extra.applications.xcca import AngularCorrelator
            n_q = 128
            n_phi = 256
            a = AngularCorrelator(n_q,n_phi)

            data = np.random.rand(n_q,n_phi)
            mask = np.random.rand(n_q,n_phi)>0.7
            ccf,ccf_mask = a.ccf(data,mask,max_order=31)
            ```
        """
        if mask is None:
            ccf = self._compute_ccf(data)
            return ccf
        else:
            ccf,ccf_mask = self._compute_ccf_masked(data,mask,max_order = max_order)
            return ccf,ccf_mask


class _CumulativeVarianceBase:
    '''
    Base class for cumulative variance computations.
    This class should never be instanciated directly.

    Attributes:
        mean: Mean value of the seen data.
        count (NDArray): Number of seen unmasked data points.
        variance: Variance of the seen data.
        bessels_correction (bool): Whether or not to apply [bessels_correction](https://en.wikipedia.org/wiki/Bessel%27s_correction){target=_blank} when accessing `self.variance`.
        no_data_to_nan (bool): Whether or not to set the mean where no data has been seen to np.nan (otherwise it is 0).
    '''
    def __init__(self,mean:NDArray=None,count:NDArray=None,m2:NDArray=None,bessels_correction:bool=False,no_data_to_nan:bool=True):
        self.bessels_correction = bessels_correction
        self.no_data_to_nan = no_data_to_nan
        if (not isinstance(mean,np.ndarray)) or (not isinstance(count,np.ndarray)) or (not isinstance(m2,np.ndarray)):
            self.count = np.array(0)
            self._mean = np.array([np.nan])
            self.m2 = np.array([np.nan])
            self.workspace = None
        else:
            self.count = count
            self._mean = mean
            self.m2 = m2
            self.workspace = np.zeros_like(mean)
        
    def _create_workspace(self,data:NDArray)->None:
        self.workspace = np.zeros_like(data)
        self._mean = np.zeros_like(data)
        self.m2 = np.zeros_like(data)

    @classmethod    
    def from_dataset(cls,*data:ArrayLike,axis=0) -> Self:
        '''
        Creates object from an array(dataset) calculating var and mean along a specified axis.
        '''
        obj = cls()
        new_data = tuple(np.moveaxis(d,axis,0) for d in data)
        for args in zip(*new_data):
            obj.update(*args)
            obj.update(*args)
        return obj
    
    def update(self,*args) -> Self:
        pass
        
    def merge(self,var:Self) -> Self:
        '''
        Merge data from other class instance into this instance.

        Args:
            var: Other instance of _CumulativeVarianceBase.
        Returns:
            Merged instance.
        '''
        return self.merge_from_data(var._mean,var.count,var.m2)
        
    def merge_from_data(self,mean:NDArray,count:NDArray,m2:NDArray)-> Self:
        pass
    
    @property
    def variance(self) -> NDArray:
        count = self.count
        out = self.m2.copy()
        out[count == 0] = np.nan
        out[count == 1] = 0
        mask = count>1
        if self.bessels_correction:
            np.divide(out,count-1,where=mask,out=out)
        else:
            np.divide(out,count,where=mask,out=out)
        return out
    @property
    def mean(self) -> NDArray:
        mean = self._mean.copy()
        if self.no_data_to_nan:
            mean[self.count==0]=np.nan
        return mean
    @property
    def data(self) ->tuple((NDArray,NDArray,NDArray)):
        return (self._mean,self.count,self.m2)
    def copy(self):
        return CumulativeVariance(mean = np.array(self.mean),count = np.array(self.count) ,m2=np.array(self.m2))

class CumulativeVarianceMasked(_CumulativeVarianceBase):
    '''
    Allows to computes the variance incrementally. 
    Slightly modified version of Welford's online algorithm, to allow computation for masked data:
    Algorithm taken from wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    '''    
    def update(self,val:NDArray,mask:NDArray[np.bool])->Self:
        """Update variance with a single data point and mask.
        
        Args:
            val (NDArray): New data point.
            mask (NDArray[np.bool]): Mask of the new data point.
        
        Returns:
            CumulativeVarianceMasked: self
        
        Examples:
            ```py
            import numpy as np
            from EXtra.utils.xcca import CumulativeVarianceMasked
        
            data = np.random.rand(2, 100, 100)
            mask = np.random.rand(2, 100, 100) > 0.7
            cv = CumulativeVarianceMasked()
            cv.update(data[0], mask=mask[0])
            cv.update(data[1], mask=mask[1])
            ```
        """
        if not isinstance(self.workspace,np.ndarray):
            self._create_workspace(val)
            self.count = np.zeros(val.shape,int)
        w = self.workspace
        # updates the running mean and variance by a single new value
        np.add(self.count,mask.astype(int),out=self.count)
        delta = mask*(val - self._mean)
        nzero_mask = self.count>0
        w[~nzero_mask]=0
        np.divide(delta,self.count.astype(float),where=nzero_mask,out=w)
        np.add(self._mean,w,out=self._mean)
        delta2 = mask*val - self._mean
        np.add(self.m2,(delta * delta2.conj()).real,out=self.m2)
        return self            
    def merge_from_data(self,mean:NDArray,count:NDArray,m2:NDArray)->Self:
        """Merge with variance from another dataset.
        
        Args:
            mean (NDArray): Mean of the other dataset.
            count (NDArray): Count of observations in the other dataset.
            m2 (NDArray): Sum of squared deviations from the mean for the other dataset.
        
        Returns:
            Self: self
        
        Examples:
            ```py
            import numpy as np
            from EXtra.utils.xcca import CumulativeVarianceMasked
        
            data = np.random.rand(100, 100, 100)
            mask = np.random.rand(100, 100, 100) > 0.7
            cv1 = CumulativeVarianceMasked.form_dataset(data[:50], mask=mask[:50])
            cv2 = CumulativeVarianceMasked.form_dataset(data[50:], mask=mask[50:])
        
            # cv1.merge(cv2)  # Same as the following line
            cv1.merge_from_data(cv1._mean, cv1.count, cv1.m2)
            ```
        """        
        # merges the data of another CummulativeVariance instance to create the combined average and variance.
        count_a = np.array(self.count)
        np.add(self.count,count,out=self.count)
        delta = mean-self._mean
        nzero_mask = self.count>0
        count = count.astype(float)
        count[nzero_mask]/=self.count[nzero_mask] 
        temp = delta*count
        np.add(self._mean,temp,out=self._mean)
        np.add(self.m2,m2 + (delta*count_a*temp.conj()).real,out = self.m2)
        return self
    
class CumulativeVariance(_CumulativeVarianceBase):
    '''
    Allows to computes the variance incrementally. 
    Welford's online algorithm:
    Algorithm taken from wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    '''        
    def update(self,val:NDArray)->Self:
        """Update variance by a single datapoint.

        Args:
            val: New data point.

        Returns:
            updated instance

        Examples:
            ```py
            import numpy as np
            from EXtra.utils.xcca import CumulativeVarianceMasked

            data = np.random.rand(2,100,100)
            cv = CumulativeVarianceMasked()
            cv.update(data[0])
            cv.update(data[1])
            ```
        """
        if self.workspace is None:
            self._create_workspace(val)
        # updates the running mean and variance by a single new value
        self.count += 1
        delta = val - self._mean
        np.add(self._mean, delta/self.count ,out=self._mean)
        delta2 = val - self._mean
        np.add(self.m2,(delta * delta2.conj()).real,out=self.m2)
        return self        
    def merge_from_data(self,mean:NDArray,count:NDArray,m2:NDArray)->Self:
        """Merge with variance from other dataset.

        Args:
            mean: Mean of the other dataset.
            count: Count of observations in the other dataset.
            m2: Sum of squared deviations from the mean for the other dataset.
        Returns:
            merged instance.
        
        Examples:
            ```py
            import numpy as np
            from EXtra.utils.xcca import CumulativeVarianceMasked

            data = np.random.rand(100,100,100)
            cv1 = CumulativeVarianceMasked.form_dataset(data[:50])
            cv2 = CumulativeVarianceMasked.form_dataset(data[50:])

            #cv1.merge(cv2) # Same as the following line
            cv1.merge_from_data(cv1._mean,cv1.count,cv1.m2)
            ```
        """
        # merges the data of another CummulativeVariance instance to create the combined average and variance.
        count_a = np.array(self.count)
        np.add(self.count,count,out=self.count)
        delta = mean-self._mean
        temp = delta*(count/self.count)
        np.add(self._mean,temp,out=self._mean)
        np.add(self.m2,m2 + (delta*count_a*temp.conj()).real,out = self.m2)
        return self

class AveragedAngularCorrelationMasked(CumulativeVarianceMasked):
    r"""
    Helper class to easily compute averages of angular cross-correlations or their Fourier coefficients.
    This class supports masked scattering data.
    

    Attributes:
        n_radial_samples (int): Number of radial sampling points.
        n_angular_samples (int): Number of uniform angular sampling points.
        max_order (int|None): Maximum considered harmonic expansion order (default = `n_angular_samples//2`) setting lower values saves RAM.
        compute_coefficients (bool): Whether to compute the harmonic coefficients of the average cross-correlation function or the average function itself.
        use_cuda (bool): Whether or not to use cuda for ffts.
        ac (AngularCorrelator): AngularCorrelator instance.
    """
    def __init__(self,
                 n_radial_samples:int=256,
                 n_angular_samples:int=1024,
                 max_order:None|int = None,
                 compute_coefficients:bool = True,
                 use_cuda:bool=False,
                 **kwargs):
        self._max_order = max_order
        self._compute_coefficients = compute_coefficients
        self.ac = AngularCorrelator(n_radial_samples,n_angular_samples,use_cuda=use_cuda)
        if compute_coefficients:
            self.process_data = self.ac.ccn
        else:
            self.process_data = self.ac.ccf
            
        super().__init__(**kwargs)

    @property
    def max_order(self):
        # Hiding max_order behind property since it should not be changed after instanciation.
        return self._max_order

    @property
    def compute_coefficients(self):
        # Hiding compute_coefficients behind property since it should not be changed after instanciation.
        return self._compute_coefficients

    @classmethod    
    def from_dataset(cls,*data:ArrayLike,axis=0,max_order=None,compute_coefficients=True,use_cuda=False) -> Self:       
        r''' Creates instance from a dataset calculating var and mean along a specified axis.
        
        Args:
            data (tuple(NDArray[np.float64],NDArray[bool]): (scattering patterns, masks)

        Returns:
            AveragedAngularCorrelationMasked instance storing mean and variance for the given data.
        '''
        new_data = tuple(np.moveaxis(d,axis,0) for d in data)
        n_q,n_phi = data[0].shape[-2:]
        obj = cls(n_q,n_phi,max_order=max_order,compute_coefficients=compute_coefficients,use_cuda=use_cuda)
        for args in zip(*new_data):
            obj.update(*args)
        return obj

    def update(self,data:NDArray[np.float64],mask:NDArray[np.bool]) -> Self:
        r''' Update average cross-correlation by a single scattering pattern.

        Args:
            data: (n_q,n_phi) Scattering pattern in polar coordinates
            mask: (n_q,n_phi) Mask for the provided scattering pattern.
        
        Returns:
            Updated instance.
        '''
        ccf,ccf_mask = self.process_data(data,mask,max_order = self.max_order)
        super().update(ccf,ccf_mask)

class AveragedAngularCorrelation(CumulativeVariance):
    r"""Helper class to make computation of averages of angular cross-correlations or their coefficients easy.

    !!! note "Unmasked data only"
        This is usefull e.g. when you have a constant mask.
        ```py
        import numpy as np
        from extra.utils.xcca import AveragedAngularCorrelation,AveragedAngularCorrelationMasked
        
        # Goal: Compute the first 31
        
        n_q,n_phi = 64,128
        max_order = 31
        scattering_patterns = np.random.rand(20,n_q,n_phi)
        constant_mask = np.random.rand(n_q,n_phi)>0.5
        
        # compute average ccf from data
        accf = AveragedAngularCorrelation(n_q,n_phi,compute_coefficients=False)
        for I in scattering_patterns:
            accf.update(I*constant_mask) 
            # Multiplying by the mask is necessary to make the mask correction work later on.
            # It ensures that masked values are all 0.
        
        # compute ccf from mask
        mask_ccf = accf.ac.ccf(constant_mask.astype(float))
        # correctd manually
        corrected_ccf,ccf_mask = accf.ac.ccf_mask_correction(accf.mean,mask_ccf)
        corrected_ccn = accf.ac.ccn_from_ccf(corrected_ccf,max_order=max_order)
        corrected_ccn_mask = np.all(ccf_mask,axis=-1)
        corrected_ccn[~corrected_ccn_mask] = np.nan
        
        # For comparison this is how you can do the same using AveragedAngularCorrelationMasked:
        accn2 = AveragedAngularCorrelationMasked(n_q,n_phi,max_order = max_order)
        for I in scattering_patterns:
            accn2.update(I,constant_mask)
        
        assert np.allclose(corrected_ccn,accn2.mean,equal_nan=True)
        ```
        The manual approach saves about 50% of computation time but you have to store the full ccf to do the manual mask correction despite only beeing interested in its 31 Fourier coefficients. AveragedAngularCorrelationMasked does the mask correction on-the-fly so the full ccf never has to be stored.
    
    Attributes:
        n_radial_samples (int): Number of radial sampling points.
        n_angular_samples (int): Number of uniform angular sampling points.
        max_order (int|None): Maximum considered harmonic expansion order (default = `n_angular_samples//2`) setting lower values saves RAM.
        compute_coefficients (bool): Whether to compute the harmonic coefficients of the average cross-correlation function or the average function itself.
        use_cuda (bool): Whether or not to use cuda for ffts.
        ac (AngularCorrelator): AngularCorrelator instance.
    """
    def __init__(self,
                 n_radial_samples=256,
                 n_angular_samples=1024,
                 max_order = None,
                 compute_coefficients = True,
                 use_cuda=False,
                 **kwargs):
        self._max_order = max_order
        self._compute_coefficients = compute_coefficients
        self.ac = AngularCorrelator(n_radial_samples,n_angular_samples,use_cuda=use_cuda)
        if compute_coefficients:
            self.process_data = self.ac.ccn
        else:
            self.process_data = self.ac.ccf
            
        super().__init__(**kwargs)

    @property
    def max_order(self):
        # Hiding max_order behind property since it should not be changed after instanciation.
        return self._max_order

    @property
    def compute_coefficients(self):
        # Hiding compute_coefficients behind property since it should not be changed after instanciation.
        return self._compute_coefficients

    @classmethod    
    def from_dataset(cls,*data:ArrayLike,axis=0,max_order=None,compute_coefficients=True,use_cuda=False) -> Self:
        r'''Creates istance from an array(dataset), calculating var and mean along a specified axis.
        
        Args:
            data (NDArray): scattering patterns.

        Returns:
            AveragedAngularCorrelation instance storing mean and variance for the given data.
        '''
        new_data = tuple(np.moveaxis(d,axis,0) for d in data)
        n_q,n_phi = data[0].shape[-2:]
        obj = cls(n_q,n_phi,max_order=max_order,compute_coefficients=compute_coefficients,use_cuda=use_cuda)
        for args in zip(*new_data):
            obj.update(*args)
        return obj
    def update(self,data):
        r''' Update average cross-correlation by a single scattering pattern.

        Args:
            data: (n_q,n_phi) Scattering pattern in polar coordinates.
        
        Returns:
            Updated instance.
        '''
        ccf = self.process_data(data,max_order = self.max_order)
        super().update(ccf)
