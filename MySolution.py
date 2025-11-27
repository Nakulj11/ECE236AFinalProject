import numpy as np
from sklearn.metrics import accuracy_score
import cvxpy as cp

# --- Task 1 ---
class MyDecentralized:
    def __init__(self, K):
        self.K = K  # number of classes
        self.W = None   # shape (K, M)
        self.b = None   # shape (K,)
        self.labels = None
        self.mean = None
        self.std = None

    def train(self, trainX, trainY):
        # Preprocessing
        self.mean = trainX.mean(axis=0)
        self.std = trainX.std(axis=0) + 1e-8
        X_norm = (trainX - self.mean) / self.std

        N, M = X_norm.shape
        self.labels = sorted(list(set(trainY)))
        label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        
        # Prepare One-vs-All labels
        # Y_binary[i, k] = 1 if sample i is class k, else -1
        Y_binary = -np.ones((N, self.K))
        for i, y in enumerate(trainY):
            Y_binary[i, label_to_idx[y]] = 1.0

        # Variables
        W = cp.Variable((self.K, M))
        b = cp.Variable(self.K)
        slack = cp.Variable((N, self.K), nonneg=True)

        # Vectorized constraints
        # Scores: (N, K)
        # We want: Y_binary * (X @ W.T + b) >= 1 - slack
        # Note: W is (K, M), X is (N, M). X @ W.T is (N, K).
        # b is (K,). Broadcasting works.
        
        scores = X_norm @ W.T + b[None, :]
        constraints = [
            cp.multiply(Y_binary, scores) >= 1 - slack
        ]

        # Objective: L1 regularization + Hinge Loss
        objective = cp.Minimize(0.001 * cp.norm1(W) + 0.001 * cp.norm1(b) + cp.sum(slack))

        problem = cp.Problem(objective, constraints)
        # Use a solver that handles large problems well if possible
        problem.solve()

        self.W = W.value
        self.b = b.value

    def predict(self, testX):
        X_norm = (testX - self.mean) / self.std
        scores = X_norm @ self.W.T + self.b
        idxs = np.argmax(scores, axis=1)
        return np.array([self.labels[i] for i in idxs])

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)
        return accuracy


##########################################################################
# --- Task 2 & Task 3 ---
##########################################################################
class MyFeatureCompression:
    def __init__(self, K):
        """
        Args:
            K (int): number of classes.
        """
        self.K = K
        # Possible bit depths for scalar quantization
        self.bit_depths = [0, 1, 2, 3, 4, 5, 6, 7, 8] 

    def _get_feature_importance(self, trainX, trainY):
        # Train a classifier to get feature importance
        clf = MyDecentralized(self.K)
        clf.train(trainX, trainY)
        # Importance: L1 norm of weights for that feature across classes
        # W is (K, M)
        return np.sum(np.abs(clf.W), axis=0)

    def _solve_allocation_lp(self, importance, B_tot, M):
        # importance: (M,)
        # B_tot: total bits
        
        # We use an LP relaxation of the bit allocation problem.
        # Variables: z[m, d] continuous in [0, 1]
        # Represents probability/fraction of assigning depth d to feature m.
        
        num_depths = len(self.bit_depths)
        z = cp.Variable((M, num_depths), nonneg=True)
        
        costs = np.array(self.bit_depths) # (D,)
        
        # Quality factors for each depth
        # Heuristic: Quality ~ 1 - 2^(-2d)
        qualities = np.array([1.0 - 2.0**(-2*d) if d > 0 else 0.0 for d in self.bit_depths])
        
        # Objective: Maximize weighted quality
        # sum_m sum_d z_{md} * importance[m] * quality[d]
        U = importance[:, None] * qualities[None, :]
        
        objective = cp.Maximize(cp.sum(cp.multiply(z, U)))
        
        constraints = [
            cp.sum(z, axis=1) == 1, # Sum of probs is 1
            cp.sum(z @ costs) <= B_tot # Total budget
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        # Rounding Scheme
        z_val = z.value
        if z_val is None:
            return np.zeros(M, dtype=int)
            
        # 1. Initial hard rounding: pick max z
        indices = np.argmax(z_val, axis=1)
        allocation = np.array([self.bit_depths[i] for i in indices])
        
        # 2. Adjust to meet budget
        current_bits = np.sum(allocation)
        
        if current_bits > B_tot:
            # Need to reduce bits.
            # Greedy removal: find feature where reducing bits hurts least (min delta Utility / delta Cost)
            # For simplicity, just reduce bits of lowest importance features until satisfied.
            # Or better: use the z values. If z was split between 2 and 3, and we picked 3, we can go to 2.
            
            # Simple heuristic: reduce bits of features with lowest importance * current_bits
            while current_bits > B_tot:
                # Candidates: features with > 0 bits
                candidates = np.where(allocation > 0)[0]
                if len(candidates) == 0:
                    break
                
                # Score: importance
                # We want to reduce bits for low importance features.
                scores = importance[candidates]
                idx_to_reduce = candidates[np.argmin(scores)]
                
                # Reduce by 1 step in bit_depths
                # Find current depth index
                current_d = allocation[idx_to_reduce]
                d_idx = self.bit_depths.index(current_d)
                new_d = self.bit_depths[d_idx - 1]
                
                allocation[idx_to_reduce] = new_d
                current_bits = np.sum(allocation)
                
        return allocation

    def _quantize_data(self, X, allocation, mins, ranges):
        # X: (N, M)
        # allocation: (M,)
        # mins: (M,)
        # ranges: (M,)
        
        X_q = np.zeros_like(X)
        
        for b in set(allocation):
            if b == 0:
                continue 
            
            mask = (allocation == b)
            n_levels = 2**b
            
            # Normalize to [0, 1]
            x_sub = (X[:, mask] - mins[mask]) / ranges[mask]
            x_sub = np.clip(x_sub, 0, 1)
            
            # Quantize
            x_int = np.round(x_sub * (n_levels - 1))
            
            # Dequantize
            x_rec = (x_int / (n_levels - 1)) * ranges[mask] + mins[mask]
            
            X_q[:, mask] = x_rec
            
        return X_q

    def run_centralized(self, trainX, trainY, valX, valY, testX, testY, B_tot_list):
        result = {'B_tot': [], 'test_accuracy': []}
        
        # 1. Get Importance
        importance = self._get_feature_importance(trainX, trainY)
        
        # 2. Get Data Stats
        mins = trainX.min(axis=0)
        maxs = trainX.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-9] = 1.0 
        
        N, M = trainX.shape
        
        for B in B_tot_list:
            # 3. Solve Allocation
            allocation = self._solve_allocation_lp(importance, B, M)
            
            # 4. Quantize
            trainX_q = self._quantize_data(trainX, allocation, mins, ranges)
            testX_q = self._quantize_data(testX, allocation, mins, ranges)
            
            # 5. Train & Evaluate
            clf = MyDecentralized(self.K)
            clf.train(trainX_q, trainY)
            acc = clf.evaluate(testX_q, testY)
            
            result['B_tot'].append(B)
            result['test_accuracy'].append(acc)
            
        return result

    def run_decentralized_per_sensor(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, k_list):
        result = {'k': [], 'test_accuracy': [], 'b_s': []}
        
        num_sensors = 4
        
        # Precompute local importances and stats
        local_importances = []
        local_stats = [] 
        
        for s in range(num_sensors):
            X_s = train_blocks[s]
            # Local importance
            clf = MyDecentralized(self.K)
            clf.train(X_s, trainY)
            imp = np.sum(np.abs(clf.W), axis=0)
            local_importances.append(imp)
            
            mins = X_s.min(axis=0)
            maxs = X_s.max(axis=0)
            rng = maxs - mins
            rng[rng < 1e-9] = 1.0
            local_stats.append((mins, rng))
        
        for k in k_list:
            # For each sensor, solve allocation with budget k
            allocations = []
            for s in range(num_sensors):
                M_s = train_blocks[s].shape[1]
                alloc = self._solve_allocation_lp(local_importances[s], k, M_s)
                allocations.append(alloc)
            
            # Quantize and Concatenate
            train_parts = []
            test_parts = []
            for s in range(num_sensors):
                mins, rng = local_stats[s]
                tr_q = self._quantize_data(train_blocks[s], allocations[s], mins, rng)
                te_q = self._quantize_data(test_blocks[s], allocations[s], mins, rng)
                train_parts.append(tr_q)
                test_parts.append(te_q)
            
            trainX_q = np.hstack(train_parts)
            testX_q = np.hstack(test_parts)
            
            # Train Fusion Center Classifier
            clf = MyDecentralized(self.K)
            clf.train(trainX_q, trainY)
            acc = clf.evaluate(testX_q, testY)
            
            result['k'].append(k)
            result['test_accuracy'].append(acc)
            result['b_s'].append(tuple([k]*4))
            
        return result

    def run_decentralized_total(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, B_tot_list):
        result = {'B_tot': [], 'test_accuracy': [], 'best_allocation': []}

        # --- PREVIOUS GLOBAL IMPLEMENTATION (Commented Out) ---
        # Explanation: If we use global importance and solve a single allocation LP for the whole image 
        # (as done below), the decentralized result matches the centralized one perfectly. 
        # This is because optimizing the sum of bits across all features globally is mathematically 
        # identical to optimizing across sensors when the sensors are just partitions of the features 
        # and we allow the allocation to be optimal globally.
        #
        # Although this solution is superior, it does not reflect a truly "decentralized" approach since it requires global knowledge.
        #
        # # Construct global training data for importance
        # trainX = np.hstack(train_blocks)
        # 
        # # Global Importance
        # importance = self._get_feature_importance(trainX, trainY)
        # 
        # # Global Stats
        # mins = trainX.min(axis=0)
        # maxs = trainX.max(axis=0)
        # ranges = maxs - mins
        # ranges[ranges < 1e-9] = 1.0
        # 
        # N, M = trainX.shape
        # 
        # # Map global indices to sensors
        # dims = [b.shape[1] for b in train_blocks]
        # offsets = np.cumsum([0] + dims)
        # 
        # for B in B_tot_list:
        #     # Solve Global Allocation
        #     allocation = self._solve_allocation_lp(importance, B, M)
        #     
        #     # Calculate per-sensor budget usage
        #     sensor_budgets = []
        #     for s in range(4):
        #         start, end = offsets[s], offsets[s+1]
        #         b_s = np.sum(allocation[start:end])
        #         sensor_budgets.append(b_s)
        #     
        #     # Quantize
        #     trainX_q = self._quantize_data(trainX, allocation, mins, ranges)
        #     
        #     # For test, we need to stack blocks first
        #     testX = np.hstack(test_blocks)
        #     testX_q = self._quantize_data(testX, allocation, mins, ranges)
        #     
        #     # Train
        #     clf = MyDecentralized(self.K)
        #     clf.train(trainX_q, trainY)
        #     acc = clf.evaluate(testX_q, testY)
        #     
        #     result['B_tot'].append(B)
        #     result['test_accuracy'].append(acc)
        #     result['best_allocation'].append(tuple(sensor_budgets))

        # --- NEW LOCAL IMPLEMENTATION ---
        # We split the total budget equally among sensors and solve locally.
        
        num_sensors = 4
        
        # Precompute local importances and stats
        local_importances = []
        local_stats = [] 
        
        for s in range(num_sensors):
            X_s = train_blocks[s]
            # Local importance
            clf = MyDecentralized(self.K)
            clf.train(X_s, trainY)
            imp = np.sum(np.abs(clf.W), axis=0)
            local_importances.append(imp)
            
            mins = X_s.min(axis=0)
            maxs = X_s.max(axis=0)
            rng = maxs - mins
            rng[rng < 1e-9] = 1.0
            local_stats.append((mins, rng))

        for B in B_tot_list:
            # Equal split
            k = B // num_sensors
            
            allocations = []
            sensor_budgets = []
            
            for s in range(num_sensors):
                M_s = train_blocks[s].shape[1]
                # Solve local allocation with budget k
                alloc = self._solve_allocation_lp(local_importances[s], k, M_s)
                allocations.append(alloc)
                sensor_budgets.append(np.sum(alloc))
            
            # Quantize and Concatenate
            train_parts = []
            test_parts = []
            for s in range(num_sensors):
                mins, rng = local_stats[s]
                tr_q = self._quantize_data(train_blocks[s], allocations[s], mins, rng)
                te_q = self._quantize_data(test_blocks[s], allocations[s], mins, rng)
                train_parts.append(tr_q)
                test_parts.append(te_q)
            
            trainX_q = np.hstack(train_parts)
            testX_q = np.hstack(test_parts)
            
            # Train Fusion Center Classifier
            clf = MyDecentralized(self.K)
            clf.train(trainX_q, trainY)
            acc = clf.evaluate(testX_q, testY)
            
            result['B_tot'].append(B)
            result['test_accuracy'].append(acc)
            result['best_allocation'].append(tuple(sensor_budgets))
            
        return result


# --- Task 3.3 ---
class MyTargetAllocator:
    def __init__(self, K):
        self.K = K

    def minimal_bits_centralized(self, feature_compressor, trainX, trainY, valX, valY, testX, testY, alpha, B_grid):
        # Outer search
        res = feature_compressor.run_centralized(trainX, trainY, valX, valY, testX, testY, B_grid)
        
        for b, acc in zip(res['B_tot'], res['test_accuracy']):
            if acc >= alpha:
                return b
        return None

    def minimal_bits_decentralized(self, feature_compressor, train_blocks, val_blocks, test_blocks, trainY, valY, testY, alpha, B_grid):
        # Outer search
        res = feature_compressor.run_decentralized_total(train_blocks, val_blocks, test_blocks, trainY, valY, testY, B_grid)
        
        for i, acc in enumerate(res['test_accuracy']):
            if acc >= alpha:
                return res['B_tot'][i], res['best_allocation'][i]
        return None, None
