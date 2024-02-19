import numpy as np
import torch
import ExtinctionModelHelper as Helper
import Dataset2D as ds

class ParallelProcessor:
    @staticmethod
    def process_parallel(model, pool, star_number, device, dtype):
        results = []
        progress_index = 0
        
        def collect_result(result):
            results.append(result)
            
            nonlocal progress_index
            progress_index += 1
            print("Progress: ", progress_index/star_number*100, "%")
            
        def error_callback(e):
            print(e)

        ell = torch.rand(star_number, device=device) * 360.
        b = torch.zeros(star_number, device=device, dtype=dtype)
        d = torch.rand(star_number, device=device) * 5.5
        K = torch.zeros(star_number, device=device, dtype=dtype)
        error = torch.zeros(star_number, device=device, dtype=dtype)

        print("Start processing")

        # Use a loop for parallel processing
        for i in range(star_number):
            pool.apply_async(
                Helper.ExtinctionModelHelper.integ_d_async,
                args=(i, Helper.ExtinctionModelHelper.compute_extinction_model_density, ell[i].data, b[i].data, d[i].data, model),
                callback=collect_result,
                error_callback=error_callback
            )
        # Close pool
        pool.close()
        pool.join()  # Wait for all processes to be completed

        # Sort the result
        print("Sorting results")
        results.sort(key=lambda x: x[0])
        K = [r for i, r in results]

        print("Adding errors")
        for i in range(star_number):
            error[i] = K[i].item() * np.random.uniform(low=0.01, high=0.1)
            K[i] = K[i].item() + np.random.normal(scale=error[i].item())

        # Return the processed dataset
        return ds.Dataset2D(ell, d, K, error)
    

    
