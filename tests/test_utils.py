import numpy as np
import pytest
import xarray as xr

from extra.utils import (
    imshow2, hyperslicer2, ridgeplot, fit_gaussian, gaussian, reorder_axes_to_shape,xcca
)


def test_imshow2():
    # Smoke test
    image = np.random.rand(100, 100)
    imshow2(image)

    image = xr.DataArray(image, dims=("foo", "bar"))
    imshow2(image)


def test_hyperslicer2():
    # Smoke test
    images = np.random.rand(10, 100, 100)
    hyperslicer2(images)


def test_ridgeplot():
    arr = xr.DataArray(np.random.normal(size=(8, 100)), dims=("foo", "bar"),
                       coords={"foo": np.arange(8) * 5})

    fig = ridgeplot(arr)
    assert len(fig.axes) == 8


def test_fit_gaussian():
    # Test with auto-generated xdata and nans/infs
    params = [0, 1, 20, 5]
    data = gaussian(np.arange(100), *params, norm=False)
    data[50] = np.nan
    data[51] = np.inf
    popt = fit_gaussian(data)
    assert np.allclose(popt, params)

    # Test with A constrained to be >0
    popt_A_pos = fit_gaussian(data, A_sign=1)
    assert np.allclose(popt_A_pos, params)

    # Test with provided xdata
    params = [0, 1, 0.5, 0.2]
    xdata = np.arange(0, 1, 0.01)
    data = gaussian(xdata, *params, norm=False)
    popt = fit_gaussian(data, xdata=xdata)
    assert np.allclose(popt, params)

    # Test passing DataArray's
    popt = fit_gaussian(xr.DataArray(data, dims=("foo",)),
                        xdata=xr.DataArray(xdata, dims=("foo",)))
    assert np.allclose(popt, params)

    # Test with a downwards-pointing peak
    params = [0, -3, 20, 5]
    data = gaussian(np.arange(100), *params, norm=False)
    popt = fit_gaussian(data, A_sign=-1)
    assert np.allclose(popt, params)

    # Test failures
    bad_params = [100, -100, 200, -200]
    assert fit_gaussian(data, p0=bad_params) is None
    popt = fit_gaussian(data, p0=bad_params, nans_on_failure=True)
    assert len(popt) == len(bad_params)
    assert np.isnan(popt).all()
    assert np.isnan(fit_gaussian(np.full(100, np.nan), nans_on_failure=True)).all()

    with pytest.raises(ValueError):
        fit_gaussian(data, p0=params[:3])


def test_reorder_axes_to_shape():
    arr = np.zeros((512, 1024, 16), dtype=np.float32)  # E.g. burst mode JUNGFRAU data
    res = reorder_axes_to_shape(arr, (16, 512, 1024))
    assert res.shape == (16, 512, 1024)
    assert res.base is arr

    res = reorder_axes_to_shape(arr, (None, 512, 1024))
    assert res.shape == (16, 512, 1024)
    assert res.base is arr

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr, (12, 512, 1024))  # Wrong dimension sizes

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr, (16, 1, 512, 1024))  # Wrong number of dimensions

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr[:, :512], (16, 512, 512))  # Ambiguous order

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr, (None, None, 1024))  # Only 1 None allowed

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr, (None, 256, 1024))  # Wildcard & wrong number

    # Check we've transposed, not reshaped
    arr = np.arange(15).reshape(3, 5)
    res = reorder_axes_to_shape(arr, (5, 3))
    assert res.shape == (5, 3)
    np.testing.assert_array_equal(res[0], [0, 5, 10])


class TestCumulativeVariance:
    'Tests for the CumulativeVariance classes.'
    def get_test_data(self,dtype,n_samples,shape,seed=12345):
        rng = np.random.default_rng(12345)
        total_shape = (n_samples,)+ shape
        if dtype == np.complex128:
            data = rng.random(total_shape)+1.j*rng.random(total_shape)
        else:
            data = rng.random(total_shape).astype(dtype)
        mask = rng.random(total_shape)>0.7
        return data,mask        
    def compute_variance_naively(self,data,mask=None,axis = 0):
        if mask is None: 
            with np.testing.suppress_warnings() as snp:
                snp.filter(RuntimeWarning) # prevents divide by zero and empty slice warnings
                mean = np.mean(data,axis=axis)
                variance = np.var(data,axis=axis)
            if data.shape[axis]==1:
                variance[:]=0
            elif data.shape[axis]<1:
                variance[:]=np.nan
        else:
            counts = np.sum(mask.astype(int),axis = axis)
            sum_ = np.sum(data*mask,axis=axis)
            sum_square = np.sum((data*data.conj())*mask,axis=axis)
            mean = np.zeros_like(sum_)
            mean_square = mean.copy()
            np.divide(sum_,counts,where = counts>0,out= mean)
            np.divide(sum_square,counts,where = counts>0,out= mean_square)
            variance = mean_square-mean*mean.conj()
    
            variance[counts==1]=0
            variance[counts==0]=np.nan
            mean[counts == 0] = np.nan
        return mean,variance
    
    def test_variance_same_as_naive_computation(self):
        '''Check if custom variance computation gives same result as naive computation via numpy'''
        dtypes = [np.float64,np.complex128]
        n_samples = [0,1,10,100]
        shapes = np.array([(0,),(1,1),(1,2),(2,1),(42,13),(14,),(2,2,2)],dtype=object)
        axes = [0,-1]
        id_grid = np.mgrid[0:len(dtypes),0:len(n_samples),0:len(shapes),0:len(axes)].reshape(4,-1)
        grid = tuple((dtypes[i],n_samples[j],shapes[k],axes[l]) for i,j,k,l in zip(*id_grid))
        
        for dtype,n,shape,ax in grid:
            data,mask = self.get_test_data(dtype,n,shape)
            #print(data.shape,n,shape,ax)
            try:
                cvm = xcca.CumulativeVarianceMasked.from_dataset(data,mask,axis = ax)
                cv = xcca.CumulativeVariance.from_dataset(data,axis = ax)
                
                mean,variance = self.compute_variance_naively(data,axis=ax)
                assert np.allclose(cv.mean,mean,equal_nan=True),"mean unequal"
                assert np.allclose(cv.variance,variance,equal_nan=True), "variances are unequal"
                
                mean,variance = self.compute_variance_naively(data,mask,axis = ax)
                assert np.allclose(cvm.mean,mean,equal_nan=True),"Masked mean unequal"
                assert np.allclose(cvm.variance,variance,equal_nan=True), "Masked variances are unequal"
            except Exception as e:
                print(f"Parameters: f{(dtype,n,shape,ax)}")
                raise e                
    def test_merge(self):
        '''Check that merging results from to data parts give same result as naive computation for entire dataset.'''
        data,mask = self.get_test_data(np.float64,100,(12,4))
        n=34
        try:
            mean,variance = self.compute_variance_naively(data)    
            cv1 = xcca.CumulativeVariance.from_dataset(data[:n])
            cv2 = xcca.CumulativeVariance.from_dataset(data[n:])
            cv1.merge(cv2)
            assert np.allclose(cv1.mean,mean),"Mean unequal after merge."
            assert np.allclose(cv1.variance,variance),"Variance unequal after merge."
            
            mean,variance = self.compute_variance_naively(data,mask)    
            cvm1 = xcca.CumulativeVarianceMasked.from_dataset(data[:n],mask[:n])
            cvm2 = xcca.CumulativeVarianceMasked.from_dataset(data[n:],mask[n:])
            cvm1.merge(cvm2)
            assert np.allclose(cvm1.mean,mean),"Masked mean unequal after merge."
            assert np.allclose(cvm1.variance,variance),"Masked variance unequal after merge."            
        except Exception as e:
            raise e   

class TestAngularCrossCorrelation:
    def get_test_data(self,n,n_q,n_phi,return_mask=False):
        rng = np.random.default_rng(12345)
        data = rng.random((n,n_q,n_phi))
        if return_mask:
            mask = rng.random((n,n_q,n_phi))>0.7
            return data,mask
        else:
            return data
        
    def test_from_dataset_same_as_update_unmasked(self):
        rng = np.random.default_rng(12345)
        N,n_q,n_phi = 25,32,64
        max_order = 11
        data = self.get_test_data(N,n_q,n_phi)
        
        ccn1 = xcca.AveragedAngularCorrelation.from_dataset(data,max_order=11)
        ccf1 = xcca.AveragedAngularCorrelation.from_dataset(data,max_order=11,compute_coefficients=False)#bug here
        
        ccn2 = xcca.AveragedAngularCorrelation(n_q,n_phi,max_order=11)
        ccf2 = xcca.AveragedAngularCorrelation(n_q,n_phi,max_order=11,compute_coefficients=False)
        for I in data:
            ccn2.update(I)
            ccf2.update(I)
            
        ccn_close = np.allclose(ccn1._mean,ccn2._mean) & np.allclose(ccn1.count,ccn2.count) & np.allclose(ccn1.m2,ccn2.m2)
        assert ccn_close, 'Unmasked: .from_dataset differs from manual updates for ccn computation.'
        ccf_close = np.allclose(ccf1._mean,ccf2._mean) & np.allclose(ccf1.count,ccf2.count) & np.allclose(ccf1.m2,ccf2.m2)
        assert ccf_close, 'Unmasked: .from_dataset differs from manual updates for ccf computation.'
        
    def test_from_dataset_same_as_update_unmasked(self):
        rng = np.random.default_rng(12345)
        N,n_q,n_phi = 25,32,64
        max_order = 11
        data,mask = self.get_test_data(N,n_q,n_phi,return_mask=True)
        
        ccn1 = xcca.AveragedAngularCorrelationMasked.from_dataset(data,mask,max_order=11)
        ccf1 = xcca.AveragedAngularCorrelationMasked.from_dataset(data,mask,max_order=11,compute_coefficients=False)
        
        ccn2 = xcca.AveragedAngularCorrelationMasked(n_q,n_phi,max_order=11)
        ccf2 = xcca.AveragedAngularCorrelationMasked(n_q,n_phi,max_order=11,compute_coefficients=False)
        for I,m in zip((data,mask)):
            ccn2.update(I,m)
            ccf2.update(I,m)
            
        ccn_close = np.allclose(ccn1._mean,ccn2._mean) & np.allclose(ccn1.count,ccn2.count) & np.allclose(ccn1.m2,ccn2.m2)
        assert ccn_close, 'Masked: .from_dataset differs from manual updates for ccn computation.'
        ccf_close = np.allclose(ccf1._mean,ccf2._mean) & np.allclose(ccf1.count,ccf2.count) & np.allclose(ccf1.m2,ccf2.m2)
        assert ccf_close, 'Masked: .from_dataset differs from manual updates for ccf computation.'
