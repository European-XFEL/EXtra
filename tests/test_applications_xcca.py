from extra.applications import xcca
import pytest
import numpy as np

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
    def get_test_data2(self,n,n_q,n_phi,snr=10,return_mask=True):
        rng = np.random.default_rng(12345)
        sample = np.ones((n_q,n_phi),float)#*np.abs(np.cos(np.linspace(0,2*np.pi,n_q)))[:,None]
        angl_dep = np.linspace(0,np.sqrt(16*np.pi),n_q)
        angl_dep *= angl_dep[::-1]
        for i,spart in enumerate(sample):
            sample[i,:n_phi//2] *= np.sin(np.linspace(0,angl_dep[i],n_phi//2))
        sample += sample[:,::-1]
    
        data = np.zeros((n,n_q,n_phi),float)
        noise = np.zeros_like(data) 
        for i in range(n):
            rot = int(rng.uniform()*n_phi)
            mean = np.roll(sample,rot,axis=-1)
            noise[i] = rng.normal(0,mean/np.sqrt(snr))
            #noise[i] = rng.normal(0,np.ones_like(mean)/np.sqrt(snr))
            data[i] = mean + noise[i]
    
        if return_mask:
            mask = rng.random(data.shape)>0.4
            mask[...,:max(1,n_phi//10)]=False
            return data,mask
        else:
            return data
    
    def direct_correlation(self,data,data2=None,mask=None,mask2=None,same_q=False):
        """
        Direct reference implementation for AngularCorrelator.ccf().
        
        If data2 is None, compute the autocorrelation of data.
        
        If same_q is False, returns correlations for all radial-bin pairs:
        shape: (n_q1, n_q2, n_phi)
        
        If same_q is True, only correlates corresponding radial bins:
        shape: (n_q, n_phi)
        """ 
        n_q1, n_phi = data.shape
        
        if data2 is None:
            data2 = data            
            if isinstance(mask, np.ndarray) and mask2 is None:
                mask2 = mask
                
        n_q2, n_phi2 = data2.shape
        
        if n_phi != n_phi2:
            raise ValueError(
                "data and data2 must have the same number of angular samples."
            )
        
        if isinstance(mask, np.ndarray) and not isinstance(mask2, np.ndarray):
            raise ValueError(
                "mask2 must be provided when mask is provided for a "
                "cross-correlation."
            )
        
        if same_q:
            corr = np.zeros((n_q1, n_phi), float)
            
            if isinstance(mask, np.ndarray):
                corr_counts = np.zeros((n_q1, n_phi), int)
                
                for i in range(n_phi):
                    tmp_corr = data * np.roll(data2, i, axis=-1)
                    tmp_mask = mask & np.roll(mask2, i, axis=-1)
                    
                    tmp_corr[~tmp_mask] = 0
                    
                    counts = np.sum(tmp_mask.astype(int), axis=-1)
                    nonzero = counts != 0
                    
                    corr[nonzero, i] = (
                        np.sum(tmp_corr, axis=-1)[nonzero] /
                        counts[nonzero]
                    )
                    
                    corr_counts[:, i] = counts
                    
                return corr, corr_counts
            
            else:
                for i in range(n_phi):
                    corr[:, i] = (
                        np.sum(
                            data * np.roll(data2, i, axis=-1),
                            axis=-1,
                        )
                        / n_phi
                    )
                    
                return corr
            
        else:
            corr = np.zeros((n_q1, n_q2, n_phi), float)
            
            if isinstance(mask, np.ndarray):
                corr_counts = np.zeros((n_q1, n_q2, n_phi), int)
                
                for i in range(n_phi):
                    tmp_corr = (
                        data[:, None, :]
                        * np.roll(data2, i, axis=-1)[None, :, :]
                    )
                    
                    tmp_mask = (
                        mask[:, None, :]
                        & np.roll(mask2, i, axis=-1)[None, :, :]
                    )
                    
                    tmp_corr[~tmp_mask] = 0
                    
                    counts = np.sum(tmp_mask.astype(int), axis=-1)
                    nonzero = counts != 0
                    
                    corr[nonzero, i] = (
                        np.sum(tmp_corr, axis=-1)[nonzero] /
                        counts[nonzero]
                    )
                    
                    corr_counts[..., i] = counts
                    
                return corr, corr_counts
            
            else:
                for i in range(n_phi):
                    corr[..., i] = (
                        np.sum(
                            data[:, None, :]
                            * np.roll(data2, i, axis=-1)[None, :, :],
                            axis=-1,
                        )
                        / n_phi
                    )
    
                return corr
    def direct_ccn_from_ccf(self, ccf):
        """Naïve Fourier transform of a CCF."""
        return np.fft.rfft(
            ccf,
            axis=-1,
            norm="forward",
        )
    
    @pytest.mark.parametrize("same_q", [False, True])
    def test_unmasked_ccf_against_naive_cross_correlation(self, same_q):
        n_q = 7
        n_phi = 32

        data1 = self.get_test_data(
            n=1,
            n_q=n_q,
            n_phi=n_phi,
        )[0]

        data2 = self.get_test_data(
            n=2,
            n_q=n_q,
            n_phi=n_phi,
        )[1]

        ac = xcca.AngularCorrelator(n_q, n_phi)

        expected = self.direct_correlation(
            data1,
            data2=data2,
            same_q=same_q,
        )

        result = ac.ccf(
            data1,
            data2=data2,
            same_q=same_q,
        )

        assert result.shape == expected.shape
        assert np.allclose(
            result,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
    @pytest.mark.parametrize("same_q", [False, True])
    def test_masked_ccf_against_naive_cross_correlation(self, same_q):
        n_q = 7
        n_phi = 32

        data1, mask1 = self.get_test_data(
            n=1,
            n_q=n_q,
            n_phi=n_phi,
            return_mask=True,
        )

        data2, mask2 = self.get_test_data(
n=2,
            n_q=n_q,
            n_phi=n_phi,
            return_mask=True,
        )

        data1 = data1[0]
        mask1 = mask1[0]

        data2 = data2[1]
        mask2 = mask2[1]

        ac = xcca.AngularCorrelator(n_q, n_phi)

        expected, expected_counts = self.direct_correlation(
            data1,
            data2=data2,
            mask=mask1,
            mask2=mask2,
            same_q=same_q,
        )

        result, result_mask = ac.ccf(
            data1,
            mask=mask1,
            data2=data2,
            mask2=mask2,
            same_q=same_q,
        )

        assert result.shape == expected.shape,f'shape mismatch: res {result.shape} expected {expected.shape}'

        assert np.allclose(
            result[result_mask],
            expected[expected_counts!=0],
            rtol=1e-12,
            atol=1e-12,
        ), f'Arrays not allclose.{same_q}'

        assert np.array_equal(
            result_mask,
            expected_counts != 0,
        ),'masks are unequal'
    @pytest.mark.parametrize("same_q", [False, True])
    def test_unmasked_ccn_against_naive_cross_correlation(self, same_q):
        n_q = 7
        n_phi = 32
        max_order = 11

        data1 = self.get_test_data(
            n=1,
            n_q=n_q,
            n_phi=n_phi,
        )[0]

        data2 = self.get_test_data(
            n=2,
            n_q=n_q,
            n_phi=n_phi,
        )[1]

        ac = xcca.AngularCorrelator(n_q, n_phi)

        expected_ccf = self.direct_correlation(
            data1,
            data2=data2,
            same_q=same_q,
        )

        expected_ccn = self.direct_ccn_from_ccf(expected_ccf)
        expected_ccn = expected_ccn[..., :max_order + 1]

        result = ac.ccn(
            data1,
            data2=data2,
            same_q=same_q,
            max_order=max_order,
        )

        assert result.shape == expected_ccn.shape

        assert np.allclose(
            result,
            expected_ccn,
            rtol=1e-11,
            atol=1e-12,
        )
    @pytest.mark.parametrize("same_q", [False, True])
    def test_masked_ccn_against_naive_cross_correlation(self, same_q):
        n_q = 7
        n_phi = 32
        max_order = 11

        data1, mask1 = self.get_test_data(
            n=1,
            n_q=n_q,
            n_phi=n_phi,
            return_mask=True,
        )

        data2, mask2 = self.get_test_data(
            n=2,
            n_q=n_q,
            n_phi=n_phi,
            return_mask=True,
        )

        data1 = data1[0]
        mask1 = mask1[0]

        data2 = data2[1]
        mask2 = mask2[1]

        ac = xcca.AngularCorrelator(n_q, n_phi)

        expected_ccf, expected_counts = self.direct_correlation(
            data1,
            data2=data2,
            mask=mask1,
            mask2=mask2,
            same_q=same_q,
        )

        expected_ccn = self.direct_ccn_from_ccf(expected_ccf)
        expected_ccn = expected_ccn[..., :max_order + 1]

        expected_ccn_mask = np.all(
            expected_counts != 0,
            axis=-1,
        )

        expected_ccn_mask = np.broadcast_to(
            expected_ccn_mask[..., None],
            expected_ccn.shape,
        )

        result, result_mask = ac.ccn(
            data1,
            mask=mask1,
            data2=data2,
            mask2=mask2,
            same_q=same_q,
            max_order=max_order,
        )

        assert result.shape == expected_ccn.shape
        assert result_mask.shape == expected_ccn_mask.shape

        assert np.allclose(
            result[result_mask],
            expected_ccn[expected_ccn_mask],
            rtol=1e-10,
            atol=1e-12,
        )

        assert np.array_equal(
            result_mask,
            expected_ccn_mask,
        )
    @pytest.mark.parametrize("same_q", [False, True])
    def test_ccn_from_ccf_matches_ccn(self, same_q):
        n_q = 7
        n_phi = 32
        max_order = 11
        data1 = self.get_test_data(
            n=1,
            n_q=n_q,
            n_phi=n_phi,
        )[0]
        data2 = self.get_test_data(
            n=2,
            n_q=n_q,
            n_phi=n_phi,
        )[1]
        ac = xcca.AngularCorrelator(n_q, n_phi)
        ccf = ac.ccf(
            data1,
            data2=data2,
            same_q=same_q,
        )
        ccn_from_ccf = ac.ccn_from_ccf(
            ccf,
            inter_correlation=not same_q,
            max_order=max_order,
        )        
        ccn = ac.ccn(
            data1,
            data2=data2,
            same_q=same_q,
            max_order=max_order,
        )
        assert np.allclose(
            ccn_from_ccf,
            ccn,
            rtol=1e-11,
            atol=1e-12,
        )
        
    def test_order_limit_does_not_change_ccn(self):
        n = 1
        n_q=33
        n_phi=64
        data,mask = self.get_test_data2(n,n_q,n_phi)
        ac = xcca.AngularCorrelator(n_q,n_phi)
        ccn_full,ccn_mask = ac.ccn(data[0],mask[0])
        for i in range(2,30):
            ccn_partial, ccn_partial_mask = ac.ccn(data[0],mask[0],max_order=i)
            assert np.allclose(ccn_full[...,:i+1],ccn_partial), f'max_order={i} differs from full computation.'
    def test_from_dataset_same_as_update_unmasked(self):
        rng = np.random.default_rng(12345)
        N,n_q,n_phi = 25,32,64
        max_order = 11
        data = self.get_test_data(N,n_q,n_phi)
        
        ccn1 = xcca.AveragedAngularCorrelation.from_dataset(data,max_order=11)
        ccf1 = xcca.AveragedAngularCorrelation.from_dataset(data,compute_coefficients=False)#bug here
        
        ccn2 = xcca.AveragedAngularCorrelation(n_q,n_phi,max_order=11)
        ccf2 = xcca.AveragedAngularCorrelation(n_q,n_phi,compute_coefficients=False)
        for I in data:
            ccn2.update(I)
            ccf2.update(I)
            
        ccn_close = np.allclose(ccn1._mean,ccn2._mean) & np.allclose(ccn1.count,ccn2.count) & np.allclose(ccn1.m2,ccn2.m2)
        assert ccn_close, 'Unmasked: .from_dataset differs from manual updates for ccn computation.'
        ccf_close = np.allclose(ccf1._mean,ccf2._mean) & np.allclose(ccf1.count,ccf2.count) & np.allclose(ccf1.m2,ccf2.m2)
        assert ccf_close, 'Unmasked: .from_dataset differs from manual updates for ccf computation.'
    def test_from_dataset_same_as_update_masked(self):
        rng = np.random.default_rng(12345)
        N,n_q,n_phi = 25,32,64
        max_order = 11
        data,mask = self.get_test_data(N,n_q,n_phi,return_mask=True)
        
        ccn1 = xcca.AveragedAngularCorrelationMasked.from_dataset(data,mask,max_order=11)
        ccf1 = xcca.AveragedAngularCorrelationMasked.from_dataset(data,mask,max_order=11,compute_coefficients=False)
        
        ccn2 = xcca.AveragedAngularCorrelationMasked(n_q,n_phi,max_order=11)
        ccf2 = xcca.AveragedAngularCorrelationMasked(n_q,n_phi,max_order=11,compute_coefficients=False)
        for I,m in zip(data,mask):
            ccn2.update(I,m)
            ccf2.update(I,m)
            
        ccn_close = np.allclose(ccn1._mean,ccn2._mean) & np.allclose(ccn1.count,ccn2.count) & np.allclose(ccn1.m2,ccn2.m2)
        assert ccn_close, 'Masked: .from_dataset differs from manual updates for ccn computation.'
        ccf_close = np.allclose(ccf1._mean,ccf2._mean) & np.allclose(ccf1.count,ccf2.count) & np.allclose(ccf1.m2,ccf2.m2)
        assert ccf_close, 'Masked: .from_dataset differs from manual updates for ccf computation.'
        
        
    @pytest.mark.parametrize("same_q", [False, True])
    @pytest.mark.parametrize("compute_coefficients", [False, True])
    def test_averaged_cross_correlation_unmasked(self,same_q,compute_coefficients):
        n = 5
        n_q = 6
        n_phi = 32
        max_order = 9
    
        data1 = self.get_test_data(
            n=n,
            n_q=n_q,
            n_phi=n_phi,
        )
    
        data2 = self.get_test_data(
            n=n,
            n_q=n_q,
            n_phi=n_phi,
        )
    
        accumulator = xcca.AveragedAngularCorrelation(
            n_radial_samples=n_q,
            n_angular_samples=n_phi,
            max_order=max_order,
            compute_coefficients=compute_coefficients,
            same_q=same_q,
            inter_correlation=True,
        )
    
        expected = []
    
        for d1, d2 in zip(data1, data2):
            ccf = self.direct_correlation(
                d1,
                data2=d2,
                same_q=same_q,
            )
    
            if compute_coefficients:
                ccf = self.direct_ccn_from_ccf(ccf)
                ccf = ccf[..., :max_order + 1]
    
            expected.append(ccf)
    
            accumulator.update(d1, data2=d2)
    
        expected = np.mean(expected, axis=0)

        assert np.allclose(
            accumulator.mean,
            expected,
            rtol=1e-10,
            atol=1e-12,
        )

    @pytest.mark.parametrize("same_q", [False, True])
    @pytest.mark.parametrize("compute_coefficients", [False, True])
    def test_averaged_cross_correlation_masked(self,same_q,compute_coefficients):
        n = 5
        n_q = 6
        n_phi = 32
        max_order = 9
    
        data1, mask1 = self.get_test_data(
            n=n,
            n_q=n_q,
            n_phi=n_phi,
            return_mask=True,
        )
    
        data2, mask2 = self.get_test_data(
            n=n,
            n_q=n_q,
            n_phi=n_phi,
            return_mask=True,
        )
    
        accumulator = xcca.AveragedAngularCorrelationMasked(
            n_radial_samples=n_q,
            n_angular_samples=n_phi,
            max_order=max_order,
            compute_coefficients=compute_coefficients,
            same_q=same_q,
            inter_correlation=True,
        )
    
        values = []
        validities = []

        for d1, m1, d2, m2 in zip(data1, mask1, data2, mask2):
            ccf, counts = self.direct_correlation(
                d1,
                data2=d2,
                mask=m1,
                mask2=m2,
                same_q=same_q,
            )
    
            valid = counts != 0
    
            if compute_coefficients:
                ccf = self.direct_ccn_from_ccf(ccf)
                ccf = ccf[..., :max_order + 1]
                valid = np.broadcast_to(
                    np.all(valid, axis=-1)[..., None],
                    ccf.shape,
                )
            else:
                valid = valid
    
            values.append(ccf)
            validities.append(valid)
    
            accumulator.update(
                d1,
                mask=m1,
                data2=d2,
                mask2=m2,
            )
    
        values = np.asarray(values)
        validities = np.asarray(validities)
    
        expected_count = np.sum(validities, axis=0)
        expected_sum = np.sum(
            np.where(validities, values, 0),
            axis=0,
        )
        
        if compute_coefficients:
            expected_mean = np.zeros_like(expected_sum, dtype=complex)
        else:
            expected_mean = np.zeros_like(expected_sum, dtype=float)
        np.divide(
            expected_sum,
            expected_count,
            where=expected_count != 0,
            out=expected_mean,
        )
        expected_mean[expected_count == 0] = np.nan
    
        assert np.allclose(
            accumulator.mean,
            expected_mean,
            equal_nan=True,
            rtol=1e-10,
            atol=1e-12,
        )
    
        assert np.array_equal(
            accumulator.count,
            expected_count,
        )
