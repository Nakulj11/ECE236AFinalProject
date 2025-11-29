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

    def train(self, trainX, trainY, valX=None, valY=None, lam=None):
        np.random.seed(67)
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

        # Hyperparameter tuning for lambda
        best_lambda = 0.001
        
        if lam is not None:
            best_lambda = lam
        elif valX is not None and valY is not None:
            # Simple grid search
            lambdas = [1e-4, 1e-3, 1e-2]
            best_acc = -1.0
            
            # We need to solve the LP for each lambda. 
            # To avoid re-creating the problem, we can use a Parameter, 
            # but for simplicity/robustness with different solvers, we'll just loop.
            # Since this is "Task 1", efficiency isn't the primary concern vs correctness.
            
            # Normalize valX
            valX_norm = (valX - self.mean) / self.std
            
            for l_val in lambdas:
                # Variables
                W_tmp = cp.Variable((self.K, M))
                b_tmp = cp.Variable(self.K)
                slack_tmp = cp.Variable((N, self.K), nonneg=True)
                
                scores_tmp = X_norm @ W_tmp.T + b_tmp[None, :]
                constraints_tmp = [cp.multiply(Y_binary, scores_tmp) >= 1 - slack_tmp]
                objective_tmp = cp.Minimize(l_val * cp.norm1(W_tmp) + l_val * cp.norm1(b_tmp) + cp.sum(slack_tmp))
                prob_tmp = cp.Problem(objective_tmp, constraints_tmp)
                prob_tmp.solve() # Let CVXPY choose the solver
                
                # Evaluate on Val
                if W_tmp.value is not None:
                    curr_W = W_tmp.value
                    curr_b = b_tmp.value
                    val_scores = valX_norm @ curr_W.T + curr_b
                    val_preds = np.argmax(val_scores, axis=1)
                    val_pred_labels = np.array([self.labels[i] for i in val_preds])
                    acc = accuracy_score(valY, val_pred_labels)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_lambda = l_val
        
        self.best_lambda = best_lambda

        # Final Train with best lambda
        W = cp.Variable((self.K, M))
        b = cp.Variable(self.K)
        slack = cp.Variable((N, self.K), nonneg=True)

        scores = X_norm @ W.T + b[None, :]
        constraints = [
            cp.multiply(Y_binary, scores) >= 1 - slack
        ]

        # Objective: L1 regularization + Hinge Loss
        objective = cp.Minimize(best_lambda * cp.norm1(W) + best_lambda * cp.norm1(b) + cp.sum(slack))

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
        
        # ILP Formulation
        # Variables: z[m, d] binary in {0, 1}
        # Represents whether we assign depth d to feature m.
        
        num_depths = len(self.bit_depths)
        z = cp.Variable((M, num_depths), boolean=True)
        
        costs = np.array(self.bit_depths) # (D,)
        
        # Quality factors for each depth
        # Heuristic: Quality ~ 1 - 2^(-2d)
        qualities = np.array([1.0 - 2.0**(-2*d) if d > 0 else 0.0 for d in self.bit_depths])
        
        # Objective: Maximize weighted quality
        # sum_m sum_d z_{md} * importance[m] * quality[d]
        U = importance[:, None] * qualities[None, :]
        
        objective = cp.Maximize(cp.sum(cp.multiply(z, U)))
        
        constraints = [
            cp.sum(z, axis=1) == 1, # Exactly one depth per feature
            cp.sum(z @ costs) <= B_tot # Total budget
        ]
        
        prob = cp.Problem(objective, constraints)
        # Use a solver that supports integer variables (e.g., GLPK_MI, CBC, SCIP, or ECOS_BB)
        # If no MIP solver is installed, this might fail or fallback. 
        # CPLEX, GUROBI, MOSEK are best if available.
        prob.solve()
        
        # Extract allocation directly from integer solution
        z_val = z.value
        if z_val is None:
            # Fallback if solver fails
            return np.zeros(M, dtype=int)
            
        indices = np.argmax(z_val, axis=1)
        allocation = np.array([self.bit_depths[i] for i in indices])
                
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
            
            # Normalize to [0, 1] using FIXED ranges (from training)
            x_sub = (X[:, mask] - mins[mask]) / ranges[mask]
            x_sub = np.clip(x_sub, 0, 1)
            
            # Quantize
            x_int = np.round(x_sub * (n_levels - 1))
            
            # Dequantize
            x_rec = (x_int / (n_levels - 1)) * ranges[mask] + mins[mask]
            
            X_q[:, mask] = x_rec
            
        return X_q

    def run_centralized(self, trainX, trainY, valX, valY, testX, testY, B_tot_list, best_lambda=0.001):
        np.random.seed(67)
        result = {'B_tot': [], 'test_accuracy': [], 'val_accuracy': []}
        
        # 1. Get Importance
        importance = self._get_feature_importance(trainX, trainY)
        
        # 2. Get Data Stats (from Training Data ONLY)
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
            valX_q = self._quantize_data(valX, allocation, mins, ranges)
            testX_q = self._quantize_data(testX, allocation, mins, ranges)
            
            # 5. Train & Evaluate
            clf = MyDecentralized(self.K)
            clf.train(trainX_q, trainY, lam=best_lambda) # No validation tuning inside the loop to save time, or pass valX_q/valY if desired
            
            val_acc = clf.evaluate(valX_q, valY)
            test_acc = clf.evaluate(testX_q, testY)
            
            result['B_tot'].append(B)
            result['test_accuracy'].append(test_acc)
            result['val_accuracy'].append(val_acc)
            
        return result

    def run_decentralized_per_sensor(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, k_list, best_lambda=0.001):
        np.random.seed(67)
        result = {'k': [], 'test_accuracy': [], 'b_s': []}
        
        num_sensors = 4
        
        # Precompute local importances and stats
        local_importances = []
        local_stats = [] 
        
        for s in range(num_sensors):
            X_s = train_blocks[s]
            # Local importance
            clf = MyDecentralized(self.K)
            clf.train(X_s, trainY, lam=best_lambda)
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
            clf.train(trainX_q, trainY, lam=best_lambda)
            acc = clf.evaluate(testX_q, testY)
            
            result['k'].append(k)
            result['test_accuracy'].append(acc)
            result['b_s'].append(tuple([k]*4))
            
        return result

    def run_decentralized_total(self, train_blocks, val_blocks, test_blocks, trainY, valY, testY, B_tot_list, best_lambda=0.001):
        np.random.seed(67)
        result = {'B_tot': [], 'test_accuracy': [], 'val_accuracy': [], 'best_allocation': []}

        # --- OPTIMIZED DECENTRALIZED ALLOCATION ---
        # We want to find the optimal allocation (b1, b2, b3, b4) such that sum(bi) <= B_tot.
        # This is equivalent to solving the allocation problem over the union of all features,
        # but respecting the sensor boundaries (which is trivial since sensors are just partitions).
        #
        # Implementation:
        # 1. Concatenate all features to form a "virtual" global view for allocation purposes.
        # 2. Compute importance for all features (or use per-sensor importance and concatenate).
        #    Using per-sensor importance is more "decentralized" in spirit (each sensor computes its own importance).
        # 3. Solve the ILP for the total budget B_tot over all M features.
        # 4. The resulting allocation automatically tells us how many bits each sensor gets.
        
        num_sensors = 4
        
        # 1. Compute Local Importances & Stats
        local_importances = []
        local_stats = []
        
        for s in range(num_sensors):
            X_s = train_blocks[s]
            clf = MyDecentralized(self.K)
            clf.train(X_s, trainY, lam=best_lambda)
            imp = np.sum(np.abs(clf.W), axis=0)
            local_importances.append(imp)
            
            mins = X_s.min(axis=0)
            maxs = X_s.max(axis=0)
            rng = maxs - mins
            rng[rng < 1e-9] = 1.0
            local_stats.append((mins, rng))
            
        # 2. Concatenate Importances for Global Allocation
        global_importance = np.concatenate(local_importances)
        total_M = len(global_importance)
        
        # Map global indices back to sensors
        dims = [b.shape[1] for b in train_blocks]
        offsets = np.cumsum([0] + dims)
        
        for B in B_tot_list:
            # 3. Solve Allocation for Total Budget
            # This implicitly finds the optimal (b1, b2, b3, b4)
            global_allocation = self._solve_allocation_lp(global_importance, B, total_M)
            
            # Split allocation back to sensors
            allocations = []
            sensor_budgets = []
            for s in range(num_sensors):
                start, end = offsets[s], offsets[s+1]
                alloc_s = global_allocation[start:end]
                allocations.append(alloc_s)
                sensor_budgets.append(np.sum(alloc_s))
            
            # 4. Quantize
            train_parts = []
            val_parts = []
            test_parts = []
            for s in range(num_sensors):
                mins, rng = local_stats[s]
                tr_q = self._quantize_data(train_blocks[s], allocations[s], mins, rng)
                va_q = self._quantize_data(val_blocks[s], allocations[s], mins, rng)
                te_q = self._quantize_data(test_blocks[s], allocations[s], mins, rng)
                train_parts.append(tr_q)
                val_parts.append(va_q)
                test_parts.append(te_q)
            
            trainX_q = np.hstack(train_parts)
            valX_q = np.hstack(val_parts)
            testX_q = np.hstack(test_parts)
            
            # 5. Train Fusion Center Classifier
            clf = MyDecentralized(self.K)
            clf.train(trainX_q, trainY, lam=best_lambda)
            
            val_acc = clf.evaluate(valX_q, valY)
            test_acc = clf.evaluate(testX_q, testY)
            
            result['B_tot'].append(B)
            result['test_accuracy'].append(test_acc)
            result['val_accuracy'].append(val_acc)
            result['best_allocation'].append(tuple(sensor_budgets))
            
        return result


# --- Task 3.3 ---
class MyTargetAllocator:
    def __init__(self, K):
        self.K = K

    def minimal_bits_centralized(self, feature_compressor, trainX, trainY, valX, valY, testX, testY, alpha, B_grid):
        # Outer search
        res = feature_compressor.run_centralized(trainX, trainY, valX, valY, testX, testY, B_grid)
        
        # Use VALIDATION accuracy to select budget
        best_B = None
        for b, acc in zip(res['B_tot'], res['val_accuracy']):
            if acc >= alpha:
                best_B = b
                break
        return best_B

    def minimal_bits_decentralized(self, feature_compressor, train_blocks, val_blocks, test_blocks, trainY, valY, testY, alpha, B_grid):
        # Outer search
        res = feature_compressor.run_decentralized_total(train_blocks, val_blocks, test_blocks, trainY, valY, testY, B_grid)
        
        # Use VALIDATION accuracy to select budget
        best_B = None
        best_alloc = None
        for i, acc in enumerate(res['val_accuracy']):
            if acc >= alpha:
                best_B = res['B_tot'][i]
                best_alloc = res['best_allocation'][i]
                break
        return best_B, best_alloc
