import h5py

def dump(filename):
    with h5py.File(filename, "r") as fid:
        energies = fid["calibration_energies"]
        print(f"energies = [{', '.join('%0.3f' % x for x in energies)}]")
        print("data = {")
        for tof_id in range(16):
            data = fid[f"calibration_data/{tof_id}"][:, :200]
            b = '],\n'.join(', '.join('%0.3f' %x for x in y) for y in data)
            print(f"{tof_id}: [")
            print(b)
            print(f"],")
        print("}")
        for tof_id in range(16):
            A = fid[f"tof_fit_result/{tof_id}/A"]
            Aa = fid[f"tof_fit_result/{tof_id}/Aa"]
            e = fid[f"tof_fit_result/{tof_id}/energy"]
            mu = fid[f"tof_fit_result/{tof_id}/mu"]
            mu_auger = fid[f"tof_fit_result/{tof_id}/mu_auger"]
            offset = fid[f"tof_fit_result/{tof_id}/offset"]
            sigma = fid[f"tof_fit_result/{tof_id}/sigma"]
            print(f"tof_fit_result[{tof_id}] = TofFitResult(energy=np.array([{', '.join('%0.3f' % x for x in e)}]),\n"
                  f"                                        mu=np.array([{', '.join('%0.3f' % x for x in mu)}]),\n"
                  f"                                        sigma=np.array([{', '.join('%0.3f' % x for x in sigma)}]),\n"
                  f"                                        A=np.array([{', '.join('%0.3f' % x for x in A)}]),\n"
                  f"                                        Aa=np.array([{', '.join('%0.3f' % x for x in Aa)}]),\n"
                  f"                                        offset=np.array([{', '.join('%0.3f' % x for x in offset)}]),\n"
                  f"                                        mu_auger=np.array([{', '.join('%0.3f' % x for x in mu_auger)}]),\n"
                  f"                                        )")
            xgm = fid[f"calibration_mean_xgm/{tof_id}"]
            print(f"calibration_mean_xgm[{tof_id}] = np.array([{', '.join('%0.3f' % x for x in xgm)}])")
dump("example_calibration_p900485_r348.h5")

