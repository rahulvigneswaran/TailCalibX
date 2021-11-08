


from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

@dataclass
class tailcalib:
    base_engine: str = "numpy"  # Options: NumPy, PyTorch

    def generate(self, X, y, tukey_value=0.9, alpha=0.0, topk=1, extra_points=0, shuffle=True):
        """Generate new datapoints

        Args:
            X : Features
            y : Corresponding labels
            tukey_value [Hyperparameter]: Value to convert any distrubution of data into a normal distribution. Defaults to 1.0.
            alpha [Hyperparameter]: Decides how spread out the generated data is. Defaults to 0.0.
            topk [Hyperparameter]: Decides how many nearby classes should be taken into consideration for the mean and std of the newly generated data. Defaults to 1.
            extra_points [Hyperparameter]: By default the number of datapoints to be generated is decided based on the class with the maximum datapoints. This variable decides how many more extra datapoints should be generated on top of that. Defaults to 0.
            shuffle (bool, optional): Shuffles the generated and original datapoints together. Defaults to True.

        Returns:
            feat_all: Tukey transformed train data + generated datapoints
            labs_all: Corresponding labels to feat_all
            generated_points: Dict that consists of just the generated points with class label as keys.
        """        

        self.X = X
        self.y = y
        self.tukey_value = tukey_value
        self.alpha = alpha
        self.topk = topk
        self.extra_points = extra_points
        self.shuffle = shuffle

        self.sanity_checks()

        if self.base_engine == "numpy":
            import numpy as np
            return self.generate_usingNumPy()
        elif self.base_engine == "pytorch":  
            import torch
            return self.generate_usingPyTorch()
        else:
            raise Exception(f"Invalid base_engine choice - '{self.base_engine}' | Choose from: 'numpy', 'pytorch'.")
    
    def sanity_checks(self,):
        """Checks whether the type of X, y inputs matches with the chosen base_engine
        """        
        if self.base_engine == "numpy":
            import numpy as np
            assert isinstance(self.X, np.ndarray),"Base Engine is set to NumPy, so the X must be a NumPy instance!"
            assert isinstance(self.y, np.ndarray),"Base Engine is set to NumPy, so the y must be a NumPy instance!"
        else:
            import torch
            assert torch.is_tensor(self.X),"Base Engine is set to PyTorch, so the X must be a PyTorch instance!"
            assert torch.is_tensor(self.y),"Base Engine is set to PyTorch, so the y must be a PyTorch instance!"

    def generate_usingNumPy(self,):
        """Generate new datapoints using Numpy
        """  
        import scipy.spatial as sp

        feat = {}
        labs = {}  
        y_unique, y_count = np.unique(self.y, return_counts=True)
        assert len(y_unique) >= self.topk, "The 'topk' is greater than the number of uniquely available classes. Try a lesser value."
        for i in y_unique:
            feat[i] = self.X[self.y == i]
            labs[i] = np.full((feat[i].shape[0],), i)

        # Class statistics
        base_means = []
        base_covs = []
        
        for i in feat.keys():  
            base_means.append(feat[i].mean(axis=0))
            base_covs.append(np.expand_dims(self.get_cov(feat[i]),axis=0))

        base_means = np.vstack(base_means)
        base_covs = np.vstack(base_covs)       

        # Tukey's transform
        for i in feat.keys():
            feat[i] = self.tukey_transform(feat[i], self.tukey_value)

        # Distribution calibration and feature sampling
        sample_from_each = self.get_sample_count(y_count,feat.keys(), self.extra_points)

        generated_points = {}

        for i in tqdm(feat.keys()):
            if np.sum(sample_from_each[i]) == 0 and self.extra_points == 0 :
                continue

            generated_points[i] = []
            for k, x_ij in zip(sample_from_each[i], feat[i]):
                if k == 0:
                    continue
                # Getting the top k nearest classes based on l2 distance
                distances = sp.distance.cdist(base_means, np.expand_dims(x_ij, axis=0)).squeeze()
                topk_idx = np.argsort(-distances)[::-1][:self.topk]
                
                # Calibrating mean and covariance
                calibrated_mean, calibrated_cov = self.calibrate_distribution(base_means[topk_idx], base_covs[topk_idx], self.topk, x_ij, self.alpha)
                
                # Trick to avoid cholesky decomposition from failing. Look at https://juanitorduz.github.io/multivariate_normal/
                EPS = 1e-4
                calibrated_cov += (np.eye(calibrated_cov.shape[0])*EPS)

                gen = np.random.multivariate_normal(calibrated_mean, calibrated_cov,(int(k),))

                generated_points[i].append(gen)    

            generated_points[i] = np.vstack(generated_points[i])

        print("Point Generation Completed!")
        print("Don't forget to use '.convert_others()' to apply tukey transformation on validation/test data before validation/testing. Use the same 'tukey_value' as the train data.\n")
        feat_all = []
        labs_all = []

        for i in labs.keys():
            feat_all.append(feat[i])
            labs_all.append(labs[i])

        feat_all = np.vstack(feat_all)
        labs_all = np.hstack(labs_all)

        for i in generated_points.keys():
            feat_all = np.concatenate((feat_all, generated_points[i]))
            labs_all = np.concatenate((labs_all, np.full((generated_points[i].shape[0],), int(i))))

        feat_all, labs_all = self.shuffle_all(feat_all, labs_all)
        return feat_all, labs_all, generated_points

    def generate_usingPyTorch(self,):
        """Generate new datapoints using PyTorch
        """  
        import torch
        device = self.X.device
        
        feat = {}
        labs = {}  
        y_unique, y_count = np.unique(self.y.cpu().numpy(), return_counts=True)
        assert len(y_unique) >= self.topk, "The 'topk' is greater than the number of uniquely available classes. Try a lesser value."
        for i in y_unique:
            feat[i] = self.X[self.y == i]
            labs[i] = torch.full((feat[i].size()[0],), i).to(device)

        # Class statistics
        base_means = []
        base_covs = []
        
        for i in feat.keys():  
            base_means.append(feat[i].mean(dim=0))
            base_covs.append(self.get_cov(feat[i]).unsqueeze(dim=0))

        base_means = torch.vstack(base_means)
        base_covs = torch.vstack(base_covs)       

        # Tukey's transform
        for i in feat.keys():
            feat[i] = self.tukey_transform(feat[i], self.tukey_value)

        # Distribution calibration and feature sampling
        sample_from_each = self.get_sample_count(y_count,feat.keys(), self.extra_points)

        generated_points = {}

        for i in tqdm(feat.keys()):
            if np.sum(sample_from_each[i]) == 0 and self.extra_points == 0 :
                continue

            generated_points[i] = []
            for k, x_ij in zip(sample_from_each[i], feat[i]):
                if k == 0:
                    continue
                # Getting the top k nearest classes based on l2 distance
                distances = torch.cdist(base_means, x_ij.unsqueeze(0)).squeeze()
                topk_idx = torch.topk(-distances, k=self.topk)[1][:self.topk]
                
                # Calibrating mean and covariance
                calibrated_mean, calibrated_cov = self.calibrate_distribution(base_means[topk_idx], base_covs[topk_idx], self.topk, x_ij, self.alpha)
                
                # Trick to avoid cholesky decomposition from failing. Look at https://juanitorduz.github.io/multivariate_normal/
                EPS = 1e-4
                calibrated_cov += (torch.eye(calibrated_cov.shape[0])*EPS).to(device) 

                new_dist = torch.distributions.multivariate_normal.MultivariateNormal(calibrated_mean, calibrated_cov)

                gen = new_dist.sample((int(k),))

                generated_points[i].append(gen)    

            generated_points[i] = torch.vstack(generated_points[i])
            torch.cuda.empty_cache()

        print("Point Generation Completed!")
        print("Don't forget to use '.convert_others()' to apply tukey transformation on validation/test data before validation/testing. Use the same 'tukey_value' as the train data.\n")
        feat_all = []
        labs_all = []

        for i in labs.keys():
            feat_all.append(feat[i])
            labs_all.append(labs[i])

        feat_all = torch.vstack(feat_all)
        labs_all = torch.hstack(labs_all).to(device)

        for i in generated_points.keys():
            feat_all = torch.cat((feat_all, generated_points[i].to(device)))
            labs_all = torch.cat((labs_all, torch.full((generated_points[i].size()[0],), int(i)).to(device)))

        feat_all, labs_all = self.shuffle_all(feat_all, labs_all)
        return feat_all, labs_all, generated_points

    def convert_others(self, X, tukey_value=1.0):
        """Use for applying tukey transformation on validation/test data before validation/testing!

        Args:
            X : Features
            tukey_value [Hyperparameter]: Value to convert any distrubution of data into a normal distribution. Defaults to 1.0.

        Returns:
            Tukey transformed data
        """
        # Sanity check
        if self.base_engine == "numpy":
            import numpy as np
            assert isinstance(X, np.ndarray),"Base Engine is set to NumPy, so the X must be a NumPy instance!"
        else:
            import torch
            assert torch.is_tensor(X),"Base Engine is set to PyTorch, so the X must be a PyTorch instance!"

        return self.tukey_transform(X, tukey_value)
            
    def shuffle_all(self, x, y):
        """Force shuffle data

        Args:
            x (float Tensor): Datapoints
            y (int): Labels

        Returns:
            floatTensor, int: Return shuffled datapoints and corresponding labels
        """
        if self.base_engine == "numpy":
            index = np.random.permutation(x.shape[0])
        else:
            index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def get_cov(self, X):
        """Calculate the covariance matrix for X

        Args:
            X (torch.tensor): Features

        Returns:
            [torch.tensor]: Covariance matrix of X
        """        
        n = X.shape[0]
        if self.base_engine == "numpy":
            mu = X.mean(axis=0) 
        else:
            mu = X.mean(dim=0) 
        X = (X - mu) 

        return 1/(n-1) * (X.transpose(1, 0) @ X)  # X^TX -> feat_size x num_of_samples @ num_of_samples x feat_size -> feat_size x feat_size

    def get_sample_count(self, count, keys, extra_points):
        """Decides how many samples must be generated based on each existing train datapoints.

        Args:
            count (list): Number of samples in each class
            keys (dict.keys): Class keys
            extra_points (int): The number of samples to be generated is based on the class with maximum sample count. "extra_points" decide how much more samples must be generated on top of that.

        Returns:
            dict: dict consists that has the info as to how many samples must be generated based on each existing train datapoints.
        """        
        sample_count_dict = {}
        for i in keys:
            current = count[i]
            head = max(count)
            # head is the sample count that we must match after the generation. This can be offset by "config["pg"]["extra_points"]". In our experiments this is set to 0 as it worked better.
            num_sample = head - current + extra_points
            ratio = num_sample / current
            #  Makes sure each datapoint is being used atleast once
            new_sample_from_each = [np.floor(ratio)] * current

            # Rest of the datapoints used for generation are decided randomly
            while True:
                if sum(new_sample_from_each) == num_sample:
                    break
                idx = np.random.randint(0, current)
                new_sample_from_each[idx] += 1
            
            # Sanity checks
            assert sum(new_sample_from_each) == num_sample
            assert len(new_sample_from_each) == current

            sample_count_dict[i] = new_sample_from_each
        return sample_count_dict


    def calibrate_distribution(self, base_means, base_cov, k, x_ij, alpha=0.0):
        """Calibration of the distribution for generation. Check equation 7 and 8 from our paper - Feature Generation for Long-tail Classification. 

        Args:
            base_means (torch.tensor): List of all the means that are used for calibration.
            base_cov (torch.tensor): List of all the covariance matrices used for calibraton.
            k (int): Number of classes used for calibration.
            x_ij (torch.tensor): Datapoint chosen to be used for generation.
            alpha (float, optional): Decides the spread of the generated samples. Defaults to 0.0.

        Returns:
            torch.tensor : Calibrated mean and covariance matrix
        """        
        if self.base_engine == "numpy":
            calibrated_mean = (base_means.sum(axis=0) + x_ij)/(k+1)
            calibrated_cov = base_cov.sum(axis=0)/k + alpha
        else:
            calibrated_mean = (base_means.sum(dim=0) + x_ij)/(k+1)
            calibrated_cov = base_cov.sum(dim=0)/k + alpha

        return calibrated_mean, calibrated_cov

    def tukey_transform(self, x, lam=0.2):
        """Transforms any distribution into normal-distribution like.

        Args:
            x (torch.tensor): Features
            lam (float, optional): Adjusts how close the transformed features will be to the origin. Defaults to 0.2.

        Returns:
            torch.tensor: Normal distribution like features.
        """        
        if lam == 0:
            EPS = 1e-6
            x = x + EPS
            return x.log()
        else :
            return x**lam
    
    def sample_test(self):
        """[For internal testing only] Uses a randomly generated sample set to check if both the engines work error free.
        """        
        if self.base_engine == "numpy":
            import numpy as np
            X = np.random.rand(200,100)
            y = np.random.randint(0,10, (200,))
            feat, lab, gen = self.generate(X=X, y=y)
            print(f"Before: {np.unique(y, return_counts=True)}")
            print(f"After: {np.unique(lab, return_counts=True)}")
        else:
            import torch
            X = torch.rand((200,100))
            y = torch.randint(0,10, (200,))
            feat, lab, gen = self.generate(X=X, y=y)
            print(f"Before: {torch.unique(y, return_counts=True)}")
            print(f"After: {torch.unique(lab, return_counts=True)}")