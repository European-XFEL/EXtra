from .base import SerializableMixin
import numpy as np
from numpy.typing import NDArray

class AngularCorrelator(SerializableMixin):
    '''
    Class to compute angular cross-correlation from data and masks given on a uniform polar grid.
    '''
    def __init__(self,n_radial_samples:int=256,n_angular_samples:int=1024,use_cuda:bool=False):
        self.n_radial_samples = n_radial_samples
        self.n_angular_samples = n_angular_samples
        self.use_cuda = use_cuda

        self.ccf_workspace = np.empty((self.n_radial_samples,self.n_angular_samples),dtype = float)
        self.ccn_workspace = np.empty((self.n_radial_samples,self.n_angular_samples//2+1),dtype = complex)
        
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

    def _asdict(self):
        d = {'n_radial_samples':self.n_radial_samples,
                'n_angular_samples':self.n_angular_samples,
                'use_cuda':self.use_cuda}
        return d

    @staticmethod 
    def ccf_mask_correction(ccf_data:NDArray[np.float64], ccf_mask:NDArray[np.bool]) -> tuple((NDArray[np.float64],NDArray[np.bool])):
        """Apply mask correction to cross-correlation

        Applies mask correction to ccf computed from data*mask (ccf_data) using ccf computed from only the mask (ccf_mask).
        Correction is done according to: J Appl Crystallogr, 2024, 57, 324 (Equation 16)
        https://journals.iucr.org/j/issues/2024/02/00/yr5118/yr5118.pdf
        
        Mask convention: masked values are 0 while unmasked values are 1.
        
        Parameters
        ----------
        ccf_data : NDArray[np.float64]
            Cross-corelation computed from data*mask
        ccf_mask : NDArray[np.bool]
            Cross-correlation computed form mask
        
        Returns
        -------
        list(NDArray[np.float64],NDArray[np.bool]
            (Corrected cross-correlation, Mask of corrected cross-correlation)
        
        Examples
        --------
        FIXME: Add docs.
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

    def _ccn_from_ccf(self,ccf:NDArray[np.float64],max_order: int|None = None) -> NDArray[np.float64]:
        """Compute Fourier series coefficients of cross-correlation.

        Parameters
        ----------
        ccf : NDArray[np.float64]
            (n_q,n_q,n_phi): Cross-correlation function $C(q_1,q_2,\phi)$
        max_order : int|None
            Maximum computed Fouerier series order.

        Returns
        -------
        NDArray[np.float64]
            (n_q,n_q,n_orders): Fourier coefficients $C_n(q_1,q_2)$

        Examples
        --------
        FIXME: Add docs.

        """
        rfft = self.rfft
        if max_order is None:
            bw = ccf.shape[-1]//2 + 1
        else:
            bw = max_order+1
            
        ccn_workspace = self.ccn_workspace[0]
        ccn = np.empty(ccf.shape[:2]+(bw,),dtype = complex)
        for q1 in range(self.n_radial_samples):
            rfft(ccf[q1,q1:],axis = -1,norm='forward',out=ccn_workspace[q1:])
            ccn[q1,q1:] = ccn_workspace[q1:,:bw]
            # Fill rest by symmetry
            ccn[q1:,q1] = ccn[q1,q1:].conj()
        return ccn
    
    def _ccf_from_ccn(self,ccn:NDArray[np.complex128],max_order: int|None = None) -> NDArray[np.float64]:
        """Compute cross-correlation function from its Fourier coefficients.

        Parameters
        ----------
        ccn : NDArray[np.complex128]
            (n_q,n_q,n_orders): Fourier coefficients $C_n(q_1,q_2)$
        max_order : int|None
            maximum considered Fourier coefficient order.

        Returns
        -------
        NDArray[np.float64]
            (n_q,n_q,n_phi): Cross-correlation function $C(q_1,q_2,\phi)$
        
        Examples
        --------
        FIXME: Add docs.

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
        """Compute cross-correlation function from polar image.

        Widh data = $I(q,\phi)$ this function computes
        $C(q_1,q_2,\phi)=\mathfrac{F}\left(I_n(q_1) I_n(q_2)^*\right)$.
        
        Using the symmetry $C(q1,q2,phi) = C(q2,q1,-phi)$
        
        Parameters
        ----------
        data : NDArray[np.float64]
            (n_q,n_phi): image data on a uniform polar grid.

        Returns
        -------
        NDArray[np.float64]
            (n_q,n_q,n_phi): Cross-correlation function $C(q_1,q_2,\phi)$.

        Examples
        --------
        FIXME: Add docs.

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
        """Compute mask corrected Fourier cofficients of the cross-correlation function.

        Parameters
        ----------
        data : NDArray[np.float64]
            (n_q,n_phi): Image data on uniform polar grid.
        mask : NDArray[np.bool]
            Image mask on uniform polar grid.
        max_order : int|None
            Maximum computed Fourier series coefficient.

        Returns
        -------
        tuple(NDArray(np.complex12),NDArray[np.bool])
            ((n_q,n_q,max_order+1) , (n_q,n_q)): (Fourier coefficients $C_n(q_1,q_2)$, Mask of Fourier coefficients.)

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
        ccn_mask = np.zeros((n_q,n_q),dtype=bool)

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
            
            ccn_mask[q1,q1:] = np.prod(mask_workspace[q1:],axis = -1).astype(bool)
            rfft(ccf_workspace[2,q1:],axis = -1, norm='forward',out = ccn_workspace[2,q1:])
            ccn[q1,q1:]=ccn_workspace[2,q1:,:bw]
            
            # Use the Symmetrie C(q1,q2,phi)=C(q2,q1,-phi) which imposes
            # the symmetry Cn(q1,q2) = Cn(q2,q1)^* on its harmonic coefficents
            ccn[q1+1:,q1,:] = ccn[q1,q1+1:,:].conj()
            ccn_mask[q1+1:,q1] = ccn_mask[q1,q1+1:]
            
        return ccn, ccn_mask
    
    def _compute_ccf_masked(self,data:NDArray[np.float64],mask:NDArray[np.bool],max_order:int|None = None) -> tuple[NDArray[np.float64],NDArray[np.bool]]:
        """Compute the mask corrected cross-correlation function.

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
            ccf = self._ccf_from_ccn(ccn,max_order=max_order)
            ccf_mask = ccn_mask
        return ccf,ccf_mask

    def _compute_ccf_masked_full(self,data:NDArray[np.float64],mask:NDArray[np.bool]) -> tuple[NDArray[np.float64],NDArray[np.bool]]:
        """Compute Cross-correlation function using all harmonic orders.

        Parameters
        ----------
        data : NDArray[np.float64]
            (n_q,n_phi): Image data on uniform polar grid.
        mask : NDArray[np.bool]
            (n_q,n_phi): Image mask on uniform polar grid.
        
        Returns
        -------
        tuple(NDArray[np.float64],NDArray[np.bool])
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
    
    def ccn(self,polar_data:NDArray[np.float64], polar_mask: NDArray[np.bool]|None  = None, max_order:int|None = None) -> NDArray[np.complex128]|tuple[NDArray[np.complex128],NDArray[np.bool]]:
        """Compute Fourier coefficients of the corss-correlation funcion.
        Lowering max_order does not save computation time but simply cuts the output to the required maximum order therefore saving RAM.
        
        Parameters
        ----------
        polar_data : NDArray[np.float64]
            (n_q,n_phi): Image data on uniform polar grid.
        polar_mask : NDArray[np.bool]|None
            If the (n_q,n_phi) Image mask array is provided this
            routine automatically applies mask correction to the
            computed Fourier coefficients..
        max_order : int|None
            Maximum computed Fourier coefficient order.

        Returns
        -------
        NDArray[np.complex128]|list(NDArray[np.complex128],NDArray[np.bool])
            If mask was provided it returns both the mask corrected
            Fourier coefficients and their mask. Other wise it just
            returns the Fourier coefficients.

        Examples
        --------
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
        if polar_mask is None:
            ccf = self._compute_ccf(polar_data)
            ccn = self._ccn_from_ccf(ccf,max_order = max_order)
            return ccn
        else:
            ccn,ccn_mask = self._compute_ccn_masked(polar_data,polar_mask,max_order = max_order)
            return ccn,ccn_mask
    
    def ccf(self,polar_data:NDArray[np.float64], polar_mask:NDArray[np.bool]|None = None, max_order:int|None = None) -> NDArray[np.float64]|tuple[NDArray[np.float64],NDArray[np.bool]]:
        """Compute the corss-correlation funcion.
        Lowering max_order does not save computation time but simply cuts the output to the required maximum order therefore saving RAM.
        
        Parameters
        ----------
        polar_data : NDArray[np.float64]
            (n_q,n_phi): Image data on uniform polar grid.
        polar_mask : NDArray[np.bool]|None
            If the (n_q,n_phi) Image mask array is provided this
            routine automatically applies mask correction to the
            computed Fourier coefficients..
        max_order : int|None
            Maximum computed Fourier coefficient order.

        Returns
        -------
        NDArray[np.float64]|list(NDArray[np.float64],NDArray[np.bool])
            If mask was provided it returns both the mask corrected
            cross-correlation and its mask. Other wise it just
            returns the cross-correlation.

        Examples
        --------
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
        if polar_mask is None:
            ccf = self._compute_ccf(polar_data)
            return ccf
        else:
            ccf,ccf_mask = self._compute_ccf_masked(polar_data,polar_mask,max_order = max_order)
            return ccf,ccf_mask

class CumulativeVarianceMasked:
    '''
    Slightly modified version of
    Welford's online algorithm:
    Algorithm taken from wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    to allow for masked data.
    '''
    def __init__(self,mean=0,count=0,m2=0,bessels_correction=True):
        self.count = count
        self.mean = mean
        self.m2 = m2
        self.bessels_correction = bessels_correction
        
    @classmethod    
    def from_dataset(cls,dataset,masks = True,axis=0):
        '''
        Creates object from an array(dataset) calculating var and mean along a specified axis.
        '''
        obj = cls()
        tmp = np.moveaxis(dataset,axis,0)
        if isinstance(masks,np.ndarray):
            tmp_mask = np.moveaxis(masks,axis,0)
            for d,m in zip(tmp,tmp_mask):
                obj.update(d,mask = m)
        else:
            for d in tmp:
                obj.update(d)
        return obj
    
    def update(self,val:np.ndarray|int|float|complex,mask=True):
        # updates the running mean and variance by a single new value
        self.count += np.ones(val.shape,dtype=int)*mask
        delta = mask*(val - self.mean)
        nzero_mask = self.count>0
        delta_tmp = delta.copy()
        delta_tmp[nzero_mask] /= self.count[nzero_mask].astype(float)
        self.mean += delta_tmp
        delta2 = mask*val - self.mean
        self.m2 += (delta * delta2.conj()).real
        return self
        
    def merge(self,var):
        # merges another CumulativeVariance instance to create the combined average and variance.
        return self.merge_from_data(var.mean,var.count,var.m2)
        
    def merge_from_data(self,mean,count,m2):
        # merges the data of another CummulativeVariance instance to create the combined average and variance.
        count_a = np.array(self.count)
        self.count += count
        #delta = mean-self.mean
        mean -= self.mean
        nzero_mask = self.count>0
        count = count.astype(float)
        count[nzero_mask]/=self.count[nzero_mask] 
        #temp = delta*count
        temp = mean*count
        #self.mean = self.mean + temp
        #self.m2 = self.m2 + m2 + (delta*count_a*temp.conj()).real
        self.mean +=  temp
        self.m2 += m2 + (mean*count_a*temp.conj()).real
        return self
    
    @property
    def variance(self):
        count = self.count
        out = self.m2.copy()
        out[count == 0] = np.nan
        out[count == 1] = 0
        mask = count>1
        if self.bessels_correction:
            out[mask] /= (count[mask]-1)
        else:
            out[mask] /= count[mask]
        return out
    @property
    def data(self):
        return (self.mean,self.count,self.m2)
    def copy(self):
        return CumulativeVariance(mean = np.array(self.mean),count = self.count ,m2=np.array(self.m2))
    
class AveragedAngularCorrelation(SerializableMixin,CumulativeVarianceMasked):
    """
    Helper class to make computation of averages of angular cross-correlations or their coefficients easy.

    Examples
    --------
    
    """
    def __init__(self,*args,
                 n_radial_samples=256,
                 n_angular_samples=1024,
                 max_order = None,
                 compute_coefficients = True,
                 use_cuda=False,
                 **kwargs):
        self._max_order = max_order
        self._compute_coefficients = compute_cofficients
        self.ac = AngularCorrelator(n_radial_samples,n_angular_samples,use_cuda=use_cuda)
        if compute_coefficients:
            self.process_data = self.ac.ccn
        else:
            self.process_data = self.ac.ccf
            
        super().__init__(*args,**kwargs)

    @property
    def max_order(self):
        # Hiding max_order behind property since it should not be changed after instanciation.
        return self._max_order

    @property
    def compute_coefficients(self):
        # Hiding compute_coefficients behind property since it should not be changed after instanciation.
        return self._compute_coefficients
    
    def _asdict(self):
        out = {"n_radial_samples":self.ac.n_radial_samples,
               "n_angular_samples":self.ac.n_angular_samples,
               "use_cuda":self.ac.use_cuda,
               "max_order":self.max_order,
               "max_order":self.compute_coefficients,
               "mean":self.mean,
               "count":self.count,
               "m2":self.m2,
               "bessels_correction":self.bessels_correction}
        return out
    
    def update(self,data,mask):
        ccf,ccf_mask = self.process_data(data,mask,max_order = self.max_order)
        super().update(ccf,ccf_mask)
