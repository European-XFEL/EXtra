import numpy as np
from enum import Enum, auto
from typing import Self
from functools import partial
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
        from extra.applications.xcca import AngularCorrelator
    
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
            import cupy as cp
            import cupyx.scipy.fft as cufft
            self.xp = cp
            self.rfft = cufft.rfft
            self.irfft = cufft.irfft
        else:
            self.xp = np
            self.rfft = np.fft.rfft
            self.irfft = np.fft.irfft

        xp = self.xp
        n_q = self.n_radial_samples
        self.bw = self.n_angular_samples // 2 + 1

        self.ccf_workspace = xp.empty((3,n_q,self.n_angular_samples),dtype = float)
        self.ccn_workspace = xp.empty((4,n_q,self.bw),dtype = complex)
        self.image_workspace = xp.empty((3,n_q,self.n_angular_samples),dtype = float)
        self.fourier_workspace = xp.empty((4,n_q,self.bw),dtype = complex)
        self.mask_workspace = xp.empty((n_q,self.n_angular_samples),dtype = bool)

    # helper functions
    def _bandwidth(self, max_order: int | None) -> int:
        if max_order is None:
            return self.bw

        if not 0 <= max_order < self.bw:
            raise ValueError(
                f"max_order must be in [0, {self.bw - 1}], got {max_order}"
            )

        return max_order + 1
    def _irfft_into(self,coeff:NDArray,out:NDArray)->NDArray:
        if self.use_cuda:
            out[...] = self.irfft(coeff,n=self.n_angular_samples,axis=-1,norm="forward")
        else:
            self.irfft(coeff,n=self.n_angular_samples,axis=-1,norm="forward",out=out)
        return out
    def _rfft_into(self,func:NDArray,out:NDArray)->NDArray:
        if func.dtype != np.float64:
            func = func.astype(float)
        if self.use_cuda:
            out[...] = self.rfft(func,n=self.n_angular_samples,axis=-1,norm="forward")
        else:
            self.rfft(func,axis=-1,norm="forward",out=out)
        return out
    def _divide_into(self,a,b,mask,out):
        if self.use_cuda:
            out[mask] = a[mask]/b[mask]
        else:
            np.divide(a,b,where = mask,out=out)
        return out    
    def _fill_symmetry_ccf(self,ccf,q):
        ccf[q+1:,q,0] = ccf[q,q+1:,0]
        ccf[q+1:,q,1:] = ccf[q,q+1:,-1:0:-1]
        return ccf
    def _fill_symmetry_ccn(self,ccn,q):
        self.xp.conjugate(ccn[q,q+1:],out = ccn[q+1:,q])
        return ccn
    def ccn_from_ccf_diagonal(self,
                              ccf:NDArray[np.float64],
                              max_order: int|None = None,
                              out: NDArray[np.complex128]|None = None) -> NDArray[np.complex128]:
        r"""Compute Fourier series coefficients of cross-correlation.

        Args:
            ccf: (n_q,n_phi): Cross-correlation function $C(q_2,\phi)$.
            max_order: Maximum computed Fouerier series order. Defaults to None.

        Returns:
            NDArray[np.float64]: (n_q,n_orders): Fourier coefficients $C_n(q_1,q_2)$.
        """
        xp = self.xp
        bw = self._bandwidth(max_order)
        ccn_workspace = self.ccn_workspace[3]
        if out is None:
            out = xp.zeros((self.n_radial_samples,bw),dtype=complex)
        out[...] = self._rfft_into(ccf,ccn_workspace)[:,:bw]        
        return out
    def ccn_from_ccf_triagonal(self,
                               ccf:NDArray[np.float64],
                               max_order: int|None = None,
                               out: NDArray[np.complex128]|None = None) -> NDArray[np.complex128]:
        r"""Compute Fourier series coefficients of cross-correlation.

        Args:
            ccf: (n_q,n_q,n_phi): Cross-correlation function $C(q_1,q_2,\phi)$.
            max_order: Maximum computed Fouerier series order. Defaults to None.

        Returns:
            NDArray[np.float64]: (n_q,n_q,n_orders): Fourier coefficients $C_n(q_1,q_2)$.
        """
        xp = self.xp
        bw = self._bandwidth(max_order)
        n_q = self.n_radial_samples
        ccn_workspace = self.ccn_workspace[0]
        if out is None:
            out = xp.zeros((n_q,n_q,bw),dtype=complex)
        rfft = self._rfft_into
        fill_sym = self._fill_symmetry_ccn
        for q1 in range(n_q):
            out[q1,q1:] = rfft(ccf[q1,q1:],ccn_workspace[q1:])[:,:bw]
            fill_sym(out,q1)
        return out    
    def ccn_from_ccf_full(self,
                          ccf:NDArray[np.float64],
                          max_order: int|None = None,
                          out: NDArray[np.complex128]|None = None) -> NDArray[np.complex128]:
        r"""Compute Fourier series coefficients of cross-correlation.

        Args:
            ccf: (n_q,n_q,n_phi): Cross-correlation function $C(q_1,q_2,\phi)$.
            max_order: Maximum computed Fouerier series order. Defaults to None.

        Returns:
            NDArray[np.float64]: (n_q,n_q,n_orders): Fourier coefficients $C_n(q_1,q_2)$.
        """
        xp = self.xp
        bw = self._bandwidth(max_order)
        n_q = self.n_radial_samples
        ccn_workspace = self.ccn_workspace[0]
        if out is None:
            out = xp.zeros((n_q,n_q,bw),dtype=complex)
        rfft = self._rfft_into
        for q1 in range(n_q):
            out[q1] = rfft(ccf[q1],ccn_workspace)[:,:bw]
        return out
    def ccn_from_ccf(self,
                     ccf:NDArray[np.float64],
                     max_order: int|None = None,
                     inter_correlation = False,
                     out: NDArray[np.complex128]|None = None) -> NDArray[np.complex128]:
        """Computes harmonic coefficients of a given cross-correlation function.
        
        Args:
            ccf (NDArray[np.float64]): Input correlation function.
            max_order (int): Maximum harmonic coefficients to compute.
            inter_correlation ([type]): [description]
            out ([type]): [description]

        Returns:
            NDArray[np.complex128]: [description]
        """
        if ccf.ndim==2:
            return self.ccn_from_ccf_diagonal(
                ccf,
                max_order=max_order,
                out=out
            )
        elif inter_correlation:
            return self.ccn_from_ccf_full(
                ccf,
                max_order=max_order,
                out=out
            )
        else:
            return self.ccn_from_ccf_triagonal(
                ccf,
                max_order=max_order,
                out=out
            )
    @staticmethod 
    def ccf_mask_correction(ccf_data:NDArray[np.float64],
                            ccf_mask:NDArray[np.bool_]) -> tuple((NDArray[np.float64],NDArray[np.bool_])):
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
    

    # Unmasked ccn routines
    def _compute_ccn_from_fourier_diagonal(
        self,
        fn,
        gn,
        out = None,
        max_order: int | None = None,
    ):
        xp = self.xp
        bw = self._bandwidth(max_order)
        fn =fn[:,:bw]
        gn_conj = self.ccn_workspace[0,:,:bw]
        xp.conjugate(gn[:,:bw],out = gn_conj)
        
        if out is None:
            out = xp.zeros((self.n_radial_samples,bw),dtype=complex)
            
        xp.multiply(fn,gn_conj,out = out)
        return out
    def _compute_ccn_from_fourier_triagonal(
        self,
        fn,
        max_order: int | None = None,
        out: NDArray[np.complex128] | None = None
    ):
        n_q = self.n_radial_samples
        xp = self.xp
        bw = self._bandwidth(max_order)

        fn = fn[:,:bw]
        fn_conj = self.ccn_workspace[1,:,:bw]
        xp.conjugate(fn,out = fn_conj)
        
        if out is None:
            out = xp.zeros((n_q,n_q,bw),dtype=complex)
            
        mult = xp.multiply
        for q1 in range(n_q):
            mult(fn[q1],fn_conj[q1:],out = out[q1,q1:])
            self._fill_symmetry_ccn(out,q1)
        return  out
    def _compute_ccn_from_fourier_full(
        self,
        fn,
        gn,
        max_order: int | None = None,
        out: NDArray[np.complex128] | None = None
    ):
        n_q = self.n_radial_samples
        xp = self.xp
        bw = self._bandwidth(max_order)
        
        fn = fn[:,:bw]
        gn_conj = self.ccn_workspace[1,:,:bw]
        xp.conjugate(gn[:,:bw],out = gn_conj)
        
        if out is None:
            out = xp.zeros((n_q,n_q,bw),dtype=complex)
            
        mult = xp.multiply
        for q1 in range(n_q):
            mult(fn[q1],gn_conj[:],out = out[q1,:])
        return  out
    def _compute_ccn(self,
                     f:NDArray[np.float64],
                     g:None|NDArray[np.float64]=None,
                     same_q:bool = False,
                     max_order: int | None = None,
                     out: NDArray[np.complex128] | None = None) ->  NDArray[np.complex128]:
        
        fn = self.fourier_workspace[0]
        self._rfft_into(f,fn)
        if g is None:
            if same_q:
                ccn = self._compute_ccn_from_fourier_diagonal(fn,fn,max_order=max_order,out=out)
            else:
                ccn = self._compute_ccn_from_fourier_triagonal(fn,max_order=max_order,out=out)
        else:
            gn = self.fourier_workspace[1]
            self._rfft_into(g,gn)
            if same_q:
                ccn = self._compute_ccn_from_fourier_diagonal(fn,gn,max_order=max_order,out=out)
            else:
                ccn = self._compute_ccn_from_fourier_full(fn,gn,max_order=max_order,out=out)
        return ccn 
        
    # Unmasked ccf routines
    def _compute_ccf_from_fourier_diagonal(
        self,
        fn,
        gn,
        out = None
    ):
        
        if out is None:
            out = self.xp.zeros((self.n_radial_samples,
                                 self.n_angular_samples),
                                dtype=float)            
        ccn_workspace = self.ccn_workspace[1]
        self._compute_ccn_from_fourier_diagonal(fn,
                                                gn,
                                                out=ccn_workspace)
        self._irfft_into(ccn_workspace,out)
        return out    
    def _compute_ccf_from_fourier_triagonal(
        self,
        fn,
        out: NDArray[np.float64] | None = None
    ):
        n_q = self.n_radial_samples
        xp = self.xp

        fn_conj = self.ccn_workspace[1,:]
        xp.conjugate(fn,out = fn_conj)

        N = self.n_angular_samples
        if out is None: 
            out = xp.zeros((n_q,n_q,N),dtype=float)
        
        mult = xp.multiply
        ccn_workspace = self.ccn_workspace[0]
        for q1 in range(n_q):
            mult(fn[q1],fn_conj[q1:],out = ccn_workspace[q1:])
            self._irfft_into(ccn_workspace[q1:],out[q1,q1:])
            self._fill_symmetry_ccf(out,q1)
        return  out
    def _compute_ccf_from_fourier_full(
        self,
        fn,
        gn,
        out: NDArray[np.float64] | None = None
    ):
        n_q = self.n_radial_samples
        xp = self.xp

        gn_conj = self.ccn_workspace[1,:]
        xp.conjugate(gn,out = gn_conj)

        N = self.n_angular_samples
        if out is None: 
            out = xp.zeros((n_q,n_q,N),dtype=float)
        
        mult = xp.multiply
        ccn_workspace = self.ccn_workspace[0]
        for q1 in range(n_q):
            mult(fn[q1],gn_conj,out = ccn_workspace)
            self._irfft_into(ccn_workspace,out[q1])
        return  out
    def _compute_ccf(self,
                     f:NDArray[np.float64],
                     g:None|NDArray[np.float64]=None,
                     same_q:bool = False,
                     out: NDArray[np.float64] | None = None) ->  NDArray[np.float64]:

        fn = self.fourier_workspace[0]
        self._rfft_into(f,fn)
        if g is None:
            if same_q:
                ccf = self._compute_ccf_from_fourier_diagonal(fn,fn,out=out)
            else:
                ccf = self._compute_ccf_from_fourier_triagonal(fn,out = out)
        else:
            gn = self.fourier_workspace[1]
            self._rfft_into(g,gn)
            if same_q:
                ccf = self._compute_ccf_from_fourier_diagonal(fn,gn,out=out)
            else:
                ccf = self._compute_ccf_from_fourier_full(fn,gn,out=out)
        return ccf
    

    # masked ccf routines
    def _compute_ccf_masked_from_fourier_diagonal(
        self,
        fn,
        fmask_n,
        gn,
        gmask_n,
        out = None,
        out_mask = None
    ):        
        if out is None:
            out = self.xp.zeros((self.n_radial_samples,
                                 self.n_angular_samples),
                                dtype=float)
        if out_mask is None:
            out_mask = self.xp.zeros((self.n_radial_samples,
                                      self.n_angular_samples),
                                     dtype=bool)
            
        ccn_workspace = self.ccn_workspace[1] # 0 is used in ccn_from_fourier_diagonal
        ccn_mask_workspace = self.ccn_workspace[2]
        ccf_workspace = self.ccf_workspace[0]
        ccf_mask_workspace = self.ccf_workspace[1]

        # Compute ccn for data and mask
        self._compute_ccn_from_fourier_diagonal(fn,gn,out=ccn_workspace)
        self._compute_ccn_from_fourier_diagonal(fmask_n,gmask_n,out=ccn_mask_workspace)        
        self._irfft_into(ccn_workspace,ccf_workspace)
        self._irfft_into(ccn_mask_workspace,ccf_mask_workspace)

        # compute the boolean mask at wich ccf is defined (i.e. could be computed)
        out_mask[:]= ccf_mask_workspace>1/(2*self.n_angular_samples)
        # correct the computed image cross correlation by dividing out the mask correlation
        self._divide_into(ccf_workspace,ccf_mask_workspace,out_mask,out)
        return out,out_mask    
    def _compute_ccf_masked_from_fourier_triagonal(
        self,
        fn,
        mask_n,
        out: NDArray[np.float64] | None = None,
        out_mask: NDArray[np.bool_] | None = None
    ):
        n_q = self.n_radial_samples
        xp = self.xp

        fn_conj = self.ccn_workspace[0]
        mask_n_conj = self.ccn_workspace[1]
        xp.conjugate(fn,out = fn_conj)
        xp.conjugate(mask_n,out = mask_n_conj)

        N = self.n_angular_samples
        if out is None:
            out = xp.zeros((n_q,n_q,N),dtype=float)
        if out_mask is None:
            out_mask = xp.zeros((n_q,n_q,N),dtype=bool)

        mult = xp.multiply
        ccn_workspace = self.ccn_workspace[2]
        ccn_mask_workspace = self.ccn_workspace[3]
        ccf_workspace = self.ccf_workspace[0]
        ccf_mask_workspace = self.ccf_workspace[1]
        mask_thresh = 1/(2*N)
        for q1 in range(n_q):
            # Compute ccn for data and mask
            mult(fn[q1],fn_conj[q1:],out = ccn_workspace[q1:])
            mult(mask_n[q1],mask_n_conj[q1:],out = ccn_mask_workspace[q1:])            
            self._irfft_into(ccn_workspace[q1:],ccf_workspace[q1:])
            self._irfft_into(ccn_mask_workspace[q1:],ccf_mask_workspace[q1:])
            
            # compute the boolean mask at wich ccf is defined (i.e. could be computed)
            out_mask[q1,q1:]= ccf_mask_workspace[q1:] > mask_thresh
            self._divide_into(ccf_workspace[q1:],ccf_mask_workspace[q1:],out_mask[q1,q1:],out[q1,q1:])
            
            # fill by triagonal symmetry
            self._fill_symmetry_ccf(out,q1)
            self._fill_symmetry_ccf(out_mask,q1)
        return  out,out_mask    
    def _compute_ccf_masked_from_fourier_full(
        self,
        fn,
        fmask_n,
        gn,
        gmask_n,
        out: NDArray[np.float64] | None = None,
        out_mask: NDArray[np.bool_] | None = None
    ):
        n_q = self.n_radial_samples
        xp = self.xp
        
        gn_conj = self.ccn_workspace[0]
        gmask_n_conj = self.ccn_workspace[1]
        xp.conjugate(gn,out = gn_conj)
        xp.conjugate(gmask_n,out = gmask_n_conj)

        N = self.n_angular_samples
        if out is None:
            out = xp.zeros((n_q,n_q,N),dtype=float)
        if out_mask is None:
            out_mask = xp.zeros((n_q,n_q,N),dtype=bool)

        mult = xp.multiply
        ccn_workspace = self.ccn_workspace[2]
        ccn_mask_workspace = self.ccn_workspace[3]
        ccf_workspace = self.ccf_workspace[0]
        ccf_mask_workspace = self.ccf_workspace[1]
        mask_thresh = 1/(2*N)
        for q1 in range(n_q):
            # Compute ccn for data and mask
            mult(fn[q1],gn_conj,out = ccn_workspace)
            mult(fmask_n[q1],gmask_n_conj,out = ccn_mask_workspace)            
            self._irfft_into(ccn_workspace,ccf_workspace)
            self._irfft_into(ccn_mask_workspace,ccf_mask_workspace)
            
            # compute the boolean mask at wich ccf is defined (i.e. could be computed)
            out_mask[q1]= ccf_mask_workspace > mask_thresh
            self._divide_into(ccf_workspace,ccf_mask_workspace,out_mask[q1],out[q1])
            
        return  out,out_mask    
    def _compute_ccf_masked(self,
                            f:NDArray[np.float64],
                            f_mask:NDArray[np.bool_],
                            g: NDArray[np.float64]|None=None,
                            g_mask: NDArray[np.bool_]|None = None,
                            same_q:bool = False,
                            out: NDArray[np.float64] | None = None,
                            out_mask: NDArray[np.bool_] | None = None) ->  NDArray[np.float64]:
        xp = self.xp
        fma = self.image_workspace[0]
        mask = self.image_workspace[2]
        fn = self.fourier_workspace[0]
        fmask_n = self.fourier_workspace[1]
        if g is None:
            mask[...] = f_mask
            xp.multiply(f,mask,out = fma)
            self._rfft_into(fma,fn)
            self._rfft_into(mask,fmask_n)
            if same_q:
                ccf = self._compute_ccf_masked_from_fourier_diagonal(fn,fmask_n,fn,fmask_n,out=out,out_mask=out_mask)
            else:
                ccf = self._compute_ccf_masked_from_fourier_triagonal(fn,fmask_n,out=out,out_mask=out_mask)
        else:
            gma = self.image_workspace[1]
            gn = self.fourier_workspace[2]
            gmask_n = self.fourier_workspace[3]
                        
            mask[...] = f_mask
            xp.multiply(f,mask,out = fma)
            self._rfft_into(mask,fmask_n)
            self._rfft_into(fma,fn)
            mask[...] = g_mask
            xp.multiply(g,mask,out = gma)
            self._rfft_into(mask,gmask_n)
            self._rfft_into(gma,gn)

            if same_q:
                ccf = self._compute_ccf_masked_from_fourier_diagonal(fn,fmask_n,
                                                                     gn,gmask_n,
                                                                     out=out,out_mask=out_mask)
            else:
                ccf = self._compute_ccf_masked_from_fourier_full(fn,fmask_n,
                                                                 gn,gmask_n,
                                                                 out=out,out_mask=out_mask)
        return ccf
    
    # masked ccn routines
    def _compute_ccn_masked_from_fourier_diagonal(
        self,
        fn,
        fmask_n,
        gn,
        gmask_n,
        max_order: int | None = None,
        out = None,
        out_mask = None,
    ):
        xp = self.xp
        bw = self._bandwidth(max_order)
        if out is None:
            out = xp.zeros((self.n_radial_samples,bw),dtype = complex)
        if out_mask is None:
            out_mask = xp.zeros((self.n_radial_samples,bw),dtype = bool)
            
        ccf_workspace = self.ccf_workspace[2] # 0 and 1 are used in ccn_masked routine
        ccn_workspace = self.ccn_workspace[3] # 0-2 are used in ccn_masked routine
        mask_workspace = self.mask_workspace
        rfft = self._rfft_into
        self._compute_ccf_masked_from_fourier_diagonal(
            fn,fmask_n,
            gn,gmask_n,
            out = ccf_workspace,
            out_mask = mask_workspace            
        )

        rfft(ccf_workspace,ccn_workspace)
        out[...] = ccn_workspace[:,:bw
                                 ]
        xp.prod(mask_workspace,axis = -1,out=out_mask[:,0])
        out_mask[...] = out_mask[..., :1]
        return out[:,:bw],out_mask
    def _compute_ccn_masked_from_fourier_triagonal(
        self,
        fn,
        mask_n,
        max_order: int|None = None,
        out: NDArray[np.complex128] | None = None,
        out_mask: NDArray[np.bool_] | None = None
    ):
        n_q = self.n_radial_samples
        bw = self._bandwidth(max_order)
        xp = self.xp

        fn_conj = self.ccn_workspace[0]
        mask_n_conj = self.ccn_workspace[1]
        xp.conjugate(fn,out = fn_conj)
        xp.conjugate(mask_n,out = mask_n_conj)

        if out is None:
            out = xp.zeros((n_q,n_q,bw),dtype=complex)
        if out_mask is None:
            out_mask = xp.zeros((n_q,n_q,bw),dtype=bool)

        mult = xp.multiply
        ccn_workspace = self.ccn_workspace[2]
        ccn_mask_workspace = self.ccn_workspace[3]
        ccf_workspace = self.ccf_workspace[0]
        ccf_mask_workspace = self.ccf_workspace[1]
        ccf_workspace2 = self.ccf_workspace[2]
        mask_workspace = self.mask_workspace
        mask_thresh = 1/(2*self.n_angular_samples)
        rfft = self._rfft_into
        fill_by_symmetry = self._fill_symmetry_ccn
        for q1 in range(n_q):
            # Compute ccn for data and mask
            mult(fn[q1],fn_conj[q1:],out = ccn_workspace[q1:])
            mult(mask_n[q1],mask_n_conj[q1:],out = ccn_mask_workspace[q1:])            
            self._irfft_into(ccn_workspace[q1:],ccf_workspace[q1:])
            self._irfft_into(ccn_mask_workspace[q1:],ccf_mask_workspace[q1:])
            # compute the boolean mask at wich ccf is defined (i.e. could be computed)
            mask_workspace[q1:]= ccf_mask_workspace[q1:] > mask_thresh
            self._divide_into(ccf_workspace[q1:],ccf_mask_workspace[q1:],mask_workspace[q1:],ccf_workspace2[q1:])
            out[q1,q1:] = rfft(ccf_workspace2[q1:],ccn_workspace[q1:])[:,:bw]
            xp.prod(mask_workspace[q1:],axis=-1,out = out_mask[q1,q1:,0])
            out_mask[q1,q1:] = out_mask[q1,q1:, :1]
            
            fill_by_symmetry(out,q1)
            out_mask[q1+1:,q1] = out_mask[q1,q1+1:]
        return  out,out_mask    
    def _compute_ccn_masked_from_fourier_full(
        self,
        fn,
        fmask_n,
        gn,
        gmask_n,
        max_order: int | None = None,
        out: NDArray[np.complex128] | None = None,
        out_mask: NDArray[np.bool_] | None = None
    ):
        n_q = self.n_radial_samples
        bw = self._bandwidth(max_order)
        xp = self.xp

        gn_conj = self.ccn_workspace[0]
        gmask_n_conj = self.ccn_workspace[1]
        xp.conjugate(gn,out = gn_conj)
        xp.conjugate(gmask_n,out = gmask_n_conj)

        if out is None:
            out = xp.zeros((n_q,n_q,bw),dtype=complex)
        if out_mask is None:
            out_mask = xp.zeros((n_q,n_q,bw),dtype=bool)

        mult = xp.multiply
        ccn_workspace = self.ccn_workspace[2]
        ccn_mask_workspace = self.ccn_workspace[3]
        ccf_workspace = self.ccf_workspace[0]
        ccf_mask_workspace = self.ccf_workspace[1]
        ccf_workspace2 = self.ccf_workspace[2]
        mask_workspace = self.mask_workspace
        mask_thresh = 1/(2*self.n_angular_samples)
        rfft = self._rfft_into
        for q1 in range(n_q):
            # Compute ccn for data and mask
            mult(fn[q1],gn_conj,out = ccn_workspace)
            mult(fmask_n[q1],gmask_n_conj,out = ccn_mask_workspace)            
            self._irfft_into(ccn_workspace,ccf_workspace)
            self._irfft_into(ccn_mask_workspace,ccf_mask_workspace)
            
            # compute the boolean mask at wich ccf is defined (i.e. could be computed)
            mask_workspace[...] = ccf_mask_workspace > mask_thresh
            self._divide_into(ccf_workspace,ccf_mask_workspace,
                              mask_workspace,ccf_workspace2)
            out[q1] = rfft(ccf_workspace2,ccn_workspace)[:,:bw]
            #xp.prod(mask_workspace,axis=-1,out = out_mask[q1])
            xp.prod(mask_workspace,axis=-1,out = out_mask[q1,:,0])
            out_mask[q1,:] = out_mask[q1,:, :1]
            
        return  out,out_mask    
    def _compute_ccn_masked(self,
                            f:NDArray[np.float64],
                            f_mask:NDArray[np.bool_],
                            g: NDArray[np.float64]|None=None,
                            g_mask: NDArray[np.bool_]|None = None,
                            same_q:bool = False,
                            max_order: int | None = None,
                            out: NDArray[np.complex128] | None = None,
                            out_mask: NDArray[np.bool_] | None = None) ->  NDArray[np.complex128]:
        xp = self.xp
        fma = self.image_workspace[0]
        mask = self.image_workspace[2]
        fn = self.fourier_workspace[0]
        fmask_n = self.fourier_workspace[1]
        if g is None:
            mask[...] = f_mask
            xp.multiply(f,mask,out = fma)
            self._rfft_into(fma,fn)
            self._rfft_into(mask,fmask_n)
            if same_q:
                ccn = self._compute_ccn_masked_from_fourier_diagonal(fn,fmask_n,fn,fmask_n,max_order = max_order,out=out,out_mask=out_mask)
            else:
                ccn = self._compute_ccn_masked_from_fourier_triagonal(fn,fmask_n,max_order = max_order,out=out,out_mask=out_mask)
        else:
            gma = self.image_workspace[1]
            gn = self.fourier_workspace[2]
            gmask_n = self.fourier_workspace[3]
                        
            mask[...] = f_mask
            xp.multiply(f,mask,out = fma)
            self._rfft_into(mask,fmask_n)
            self._rfft_into(fma,fn)
            mask[...] = g_mask
            xp.multiply(g,mask,out = gma)
            self._rfft_into(mask,gmask_n)
            self._rfft_into(gma,gn)

            if same_q:
                ccn = self._compute_ccn_masked_from_fourier_diagonal(fn,fmask_n,
                                                                     gn,gmask_n,
                                                                     max_order= max_order,
                                                                     out=out,out_mask=out_mask)
            else:
                ccn = self._compute_ccn_masked_from_fourier_full(fn,fmask_n,
                                                                 gn,gmask_n,
                                                                 max_order=max_order,
                                                                 out=out,out_mask=out_mask)
        return ccn
    
    # public facing API
    def ccn(self,
            data:NDArray[np.float64],
            mask: NDArray[np.bool_]|None  = None,
            data2:NDArray[np.float64]|None = None,
            mask2: NDArray[np.bool_]|None  = None,
            same_q: bool = False,
            max_order:int|None = None,
            out: NDArray[np.complex128] | None = None,
            out_mask: NDArray[np.bool_] | None = None) -> NDArray[np.complex128]|tuple[NDArray[np.complex128],NDArray[np.bool_]]:
        r"""Compute Fourier coefficients of the corss-correlation funcion.

        Lowering max_order does not save computation time but simply cuts the output to the required maximum order therefore saving RAM.

        Args:
            data: (n_q,n_phi): Image data on uniform polar grid.
            mask: If the (n_q,n_phi) Image mask array is provided this
                routine automatically applies mask correction to the
                computed Fourier coefficients.. Defaults to None.
            data2: (n_q,n_phi): Optional second image data for inter_pattern correlation computation.
            mask2: (n_q,n_phi): Optional second image mask for inter_pattern correlation computation.
            same_q: bool: Whether to compute correlations for q1=q2 or q1!=q2.
            max_order: int: maximum computed harmonic order of the cross-correlation function.
            out: (n_q,n_q,max_order+1)|(n_q,max_order+1): Optional array to store the output ccn to. 
            out_mask: (n_q,n_q,max_order+1)|(n_q,max_order+1): Optional array to store the output ccn mask to.  
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
            ccn = self._compute_ccn(data,g = data2,same_q = same_q,max_order = max_order,out=out)
            return ccn
        else:
            ccn,ccn_mask = self._compute_ccn_masked(data,
                                                    mask,
                                                    g = data2,
                                                    g_mask = mask2,
                                                    same_q = same_q,
                                                    max_order = max_order,
                                                    out=out,
                                                    out_mask=out_mask)
            return ccn,ccn_mask        
    def ccf(self,
            data:NDArray[np.float64],
            mask:NDArray[np.bool_]|None = None,
            data2:NDArray[np.float64]|None = None,
            mask2: NDArray[np.bool_]|None  = None,
            same_q: bool = False,
            out: NDArray[np.float64] | None = None,
            out_mask: NDArray[np.bool_] | None = None) -> NDArray[np.float64]|tuple[NDArray[np.float64],NDArray[np.bool_]]:
        r"""Compute the corss-correlation funcion.

        Lowering max_order does not save computation time but simply cuts the output to the required maximum order therefore saving RAM.

        Args:
            data: (n_q,n_phi): Image data on uniform polar grid.
            mask: If the (n_q,n_phi) Image mask array is provided this
                routine automatically applies mask correction to the
                computed Fourier coefficients.. Defaults to None.
            data2:NDArray[np.float64]|None = None,
            mask2: NDArray[np.bool_]|None  = None,
            same_q: bool = False,
            max_order: Maximum computed Fourier coefficient order. Defaults to None.
            out: (n_q,n_q,n_phi)|(n_q,n_phi): Optional array to store the output ccf to. 
            out_mask: (n_q,n_q,n_phi)|(n_q,n_phi): Optional array to store the output ccf mask to.  
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
            ccf,ccf_mask = a.ccf(data,mask)
            ```
        """
        if mask is None:
            ccf = self._compute_ccf(data,g=data2,same_q=same_q,out = out)
            return ccf
        else:
            ccf,ccf_mask = self._compute_ccf_masked(data,
                                                    mask,
                                                    g = data2,
                                                    g_mask = mask2,
                                                    same_q = same_q,
                                                    out=out,
                                                    out_mask=out_mask)
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
        return type(self)(mean = np.array(self.mean),
                          count = np.array(self.count),
                          m2=np.array(self.m2),
                          bessels_correction = self.bessels_correction,
                          no_data_to_nan = self.no_data_to_nan)
class CumulativeVarianceMasked(_CumulativeVarianceBase):
    '''
    Allows to computes the variance incrementally. 
    Slightly modified version of Welford's online algorithm, to allow computation for masked data:
    Algorithm taken from wikipedia: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    '''    
    def update(self,val:NDArray,mask:NDArray[np.bool_])->Self:
        """Update variance with a single data point and mask.
        
        Args:
            val (NDArray): New data point.
            mask (NDArray[np.bool_]): Mask of the new data point.
        
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
        if self.workspace is None:
            self.count = np.array(count, copy=True)
            self._mean = np.array(mean, copy=True)
            self.m2 = np.array(m2, copy=True)
            self.workspace = np.zeros_like(self._mean)
            return self
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
        if self.workspace is None:
            self.count = np.array(count, copy=True)
            self._mean = np.array(mean, copy=True)
            self.m2 = np.array(m2, copy=True)
            self.workspace = np.zeros_like(self._mean)
            return self
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
        ac (AngularCorrelator): AngularCorrelator instance.
    """
    def __init__(self,
                 n_radial_samples:int=256,
                 n_angular_samples:int=1024,
                 max_order:None|int = None,
                 compute_coefficients:bool = True,
                 same_q = False,
                 inter_correlation = False,
                 **kwargs):
        self._max_order = max_order
        self._compute_coefficients = compute_coefficients
        self.ac = AngularCorrelator(n_radial_samples,n_angular_samples,use_cuda=False)
        self.inter_correlation = inter_correlation
        self.same_q = same_q
        if compute_coefficients:
            self.process_data = partial(self.ac.ccn,
                                        max_order = max_order,
                                        same_q=same_q)
        else:
            self.process_data = partial(self.ac.ccf,
                                        same_q=same_q)

        super().__init__(**kwargs)

    def copy(self):
        new = type(self)(n_radial_samples=self.ac.n_radial_samples,
                         n_angular_samples=self.ac.n_angular_samples,
                         max_order = self._max_order,
                         compute_coefficients = self._compute_coefficients,
                         same_q = self.same_q,
                         inter_correlation = self.inter_correlation)
        
        new.count = self.count.copy()
        new.m2 = self.m2.copy()
        new._mean = self._mean.copy()
        new.bessels_correction = self.bessels_correction
        new.no_data_to_nan = self.no_data_to_nan
        return new
    @property
    def max_order(self):
        # Hiding max_order behind property since it should not be changed after instanciation.
        return self._max_order

    @property
    def compute_coefficients(self):
        # Hiding compute_coefficients behind property since it should not be changed after instanciation.
        return self._compute_coefficients

    @classmethod    
    def from_dataset(cls,*data:ArrayLike,axis=0,max_order=None,compute_coefficients=True) -> Self:       
        r''' Creates instance from a dataset calculating var and mean along a specified axis.
        
        Args:
            data (tuple(NDArray[np.float64],NDArray[bool]): (scattering patterns, masks)

        Returns:
            AveragedAngularCorrelationMasked instance storing mean and variance for the given data.
        '''
        new_data = tuple(np.moveaxis(d,axis,0) for d in data)
        n_q,n_phi = data[0].shape[-2:]
        obj = cls(n_q,n_phi,max_order=max_order,compute_coefficients=compute_coefficients)
        for args in zip(*new_data):
            obj.update(*args)
        return obj
    def update(self,
               data:NDArray[np.float64],
               mask:NDArray[np.bool_],
               data2:NDArray[np.float64]|None=None,
               mask2:NDArray[np.bool_]|None  =None) -> Self:
        r''' Update average cross-correlation by a single scattering pattern.

        Args:
            data: (n_q,n_phi) Scattering pattern in polar coordinates
            mask: (n_q,n_phi) Mask for the provided scattering pattern.
        
        Returns:
            Updated instance.
        '''
        if self.inter_correlation:
            if data2 is None:
                raise ValueError("Inter-correlation computation requires the argument data2 to be given but data2 is None.")
            if mask2 is None:
                raise ValueError("Inter-correlation computation requires the argument mask2 to be given but mask2 is None.")
            cc,cc_mask = self.process_data(data,mask=mask,data2=data2,mask2=mask2)
        else:
            cc,cc_mask = self.process_data(data,mask)
        super().update(cc,cc_mask)
        return self
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
        ac (AngularCorrelator): AngularCorrelator instance.
    """
    def __init__(self,
                 n_radial_samples=256,
                 n_angular_samples=1024,
                 max_order = None,
                 compute_coefficients = True,
                 same_q = False,
                 inter_correlation = False,
                 **kwargs):
        self._max_order = max_order
        self._compute_coefficients = compute_coefficients
        self.ac = AngularCorrelator(n_radial_samples,n_angular_samples,use_cuda=False)
        self.inter_correlation=inter_correlation
        self.same_q = same_q
        if compute_coefficients:
            self.process_data = partial(self.ac.ccn,
                                        max_order = max_order,
                                        same_q=same_q)
        else:
            self.process_data = partial(self.ac.ccf,
                                        same_q=same_q)
                                        
            
        super().__init__(**kwargs)
        
    def copy(self):
        new = type(self)(n_radial_samples=self.ac.n_radial_samples,
                         n_angular_samples=self.ac.n_angular_samples,
                         max_order = self._max_order,
                         compute_coefficients = self._compute_coefficients,
                         same_q = self.same_q,
                         inter_correlation = self.inter_correlation)
        
        new.count = self.count.copy()
        new.m2 = self.m2.copy()
        new._mean = self._mean.copy()
        new.bessels_correction = self.bessels_correction
        new.no_data_to_nan = self.no_data_to_nan
        return new
    
    @property
    def max_order(self):
        # Hiding max_order behind property since it should not be changed after instanciation.
        return self._max_order
    
    @property
    def compute_coefficients(self):
        # Hiding compute_coefficients behind property since it should not be changed after instanciation.
        return self._compute_coefficients
    
    @classmethod    
    def from_dataset(cls,*data:ArrayLike,axis=0,max_order=None,compute_coefficients=True) -> Self:
        r'''Creates istance from an array(dataset), calculating var and mean along a specified axis.
        
        Args:
            data (NDArray): scattering patterns.

        Returns:
            AveragedAngularCorrelation instance storing mean and variance for the given data.
        '''
        new_data = tuple(np.moveaxis(d,axis,0) for d in data)
        n_q,n_phi = data[0].shape[-2:]
        obj = cls(n_q,n_phi,max_order=max_order,compute_coefficients=compute_coefficients)
        for args in zip(*new_data):
            obj.update(*args)
        return obj
    
    def update(self,data,data2 = None):
        r''' Update average cross-correlation by a single scattering pattern.

        Args:
            data: (n_q,n_phi) Scattering pattern in polar coordinates.
        
        Returns:
            Updated instance.
        '''
        if self.inter_correlation:
            if data2 is None:
                raise ValueError("Inter-correlation computation requires the argument data2 to be given but data2 is None.")
            cc = self.process_data(data,data2=data2)
        else:
            cc = self.process_data(data)
        super().update(cc)
        return self

