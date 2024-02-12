import numpy as np
import torch
import ExtinctionModelHelper as Helper
import Dataset2D

class ParallelProcessor:
    @staticmethod
    def process_parallel(model, pool, n, device, dtype):
        results = []
        
        def collect_result(result):
            global results
            results.append(result)

        ell = torch.rand(n, device=device) * 360.
        b = torch.zeros(n, device=device, dtype=dtype)
        dist = torch.rand(n, device=device) * 5.5
        K = torch.zeros(n, device=device, dtype=dtype)
        error = torch.zeros(n, device=device, dtype=dtype)

        print("Start processing")

        # Use a loop for parallel processing
        for i in range(n):
            pool.apply_async(Helper.ExtinctionModelHelper.integ_d, args=(i, Helper.ExtinctionModelHelper.compute_extinction_model_density, ell[i].data, b[i].data, dist[i].data, model),
                            callback=collect_result)

        # Close pool
        pool.close()
        pool.join()  # Wait for all processes to be completed

        # Sort the result
        print("Sorting results")
        results.sort(key=lambda x: x[0])
        K = [r for i, r in results]

        print("Adding errors")
        for i in range(n):
            error[i] = K[i].item() * np.random.uniform(low=0.01, high=0.1)
            K[i] = K[i].item() + np.random.normal(scale=error[i].item())

        # Return the processed dataset
        return Dataset2D(ell, dist, K, error)
    

    
