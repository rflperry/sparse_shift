import numpy as np
from causaldag import DAG
from sparse_shift.utils import dag2cpdag, cpdag2dags
from sparse_shift.testing import test_dag_shifts


class FullPC:
    """
    Pools all the data and computes the oracle PC algorithm result.
    """
    def __init__(self, dag):
        self.domains_ = []
        self.interv_targets_ = set()
        self.dag = dag  # adj matrix
        
    def add_environment(self, interventions):
        self.interv_targets_.update(interventions)
        self.domains_.append(interventions)
        
    def get_mec_dags(self):
        if len(self.domains_) == 1:
            return cpdag2dags(dag2cpdag(self.dag))
        else:
            intv_cpdag = dag2cpdag(self.dag, list(self.interv_targets_))
            return cpdag2dags(intv_cpdag)
    
    def get_mec_cpdag(self):
        if len(self.domains_) == 1:
            return dag2cpdag(self.dag)
        else:
            intv_cpdag = dag2cpdag(self.dag, list(self.interv_targets_))
            return intv_cpdag
    

class PairwisePC:
    """
    Oracle evaluation of the PC algorithm on all pairs of environments, orienting edges
    in the final answer if any pair orients an edges.
    """
    def __init__(self, dag):
        self.interv_targets_ = []
        self.dag = dag  # adj matrix
        self.union_cpdag_ = np.zeros(dag.shape)

    def add_environment(self, interventions):        
        for prior_targets in self.interv_targets_:
            pairwise_targets = np.unique(np.hstack((prior_targets, interventions))).astype(int)
            intv_cpdag = dag2cpdag(self.dag, pairwise_targets)
            self.union_cpdag_ += intv_cpdag
        
        self.interv_targets_.append(interventions)
        
    def get_mec_dags(self):
        if len(self.interv_targets_) == 1:
            return cpdag2dags(dag2cpdag(self.dag))
        else:
            cpdag = (self.union_cpdag_ >= np.max(self.union_cpdag_)).astype(int)
            return cpdag2dags(cpdag)
        
    def get_mec_cpdag(self):
        if len(self.interv_targets_) == 1:
            return dag2cpdag(self.dag)
        else:
            cpdag = (self.union_cpdag_ >= np.max(self.union_cpdag_)).astype(int)
            return cpdag
        

class MinChangeOracle:
    """
    Oracle test of the number of mechanism changes each DAG in a Markov equivalence
    class experiences.
    """
    def __init__(self, dag):
        self.interv_targets_ = []
        self.dag = dag  # adj matrix
        self.min_dags_ = np.asarray(cpdag2dags(dag2cpdag(dag)))
        
    def add_environment(self, interventions):
        for prior_targets in self.interv_targets_:
            n_changes = np.zeros(len(self.min_dags_))
            pairwise_targets = np.unique(np.hstack((prior_targets, interventions))).astype(int)

            n_vars = self.dag.shape[0]
            aug_dag_adj = np.zeros((n_vars+1, n_vars+1))
            aug_dag_adj[:-1, :-1] = self.dag
            aug_dag_adj[-1][pairwise_targets] = 1
            aug_dag = DAG().from_amat(aug_dag_adj)

            for i, dag in enumerate(self.min_dags_):
                n_changes[i] += self._num_changes(aug_dag, dag, pairwise_targets)
            min_idx = np.where(n_changes == min(n_changes))[0]
            self.min_dags_ = self.min_dags_[np.asarray(min_idx)]
        
        self.interv_targets_.append(interventions)
        
    def get_min_dags(self):
        return self.min_dags_
    
    def get_min_cpdag(self):
        cpdag = (np.sum(self.min_dags_, axis=0) > 0).astype(int)
        return cpdag
        
    def _num_changes(self, true_aug_dag, dag_adj, targets):
        n_vars = dag_adj.shape[0]
        d_seps = [
            true_aug_dag.dsep(n_vars, i, np.where(dag_adj.T[i] != 0)[0])
            for i in range(n_vars)
        ]
        num_changes = n_vars - np.sum(d_seps)
        return num_changes


class MinChange:
    """
    Computes the number of pairwise mechanism changes in all DAGs in a given
    Markov equivalence class across given environment datasets
    """
    def __init__(self, cpdag, test='kci', alpha=0.05, scale_alpha=True, test_kwargs={}):
        self.cpdag = cpdag
        self.test = test
        self.alpha = alpha
        self.scale_alpha = scale_alpha
        self.test_kwargs = test_kwargs
        self.dags_ = np.asarray(cpdag2dags(cpdag))
        self.n_vars_ = cpdag.shape[0]
        self.alpha_ = alpha
        if scale_alpha:
            self.alpha_ /= self.n_vars_  # account for false positive rate within dag
        self.n_envs_ = 0
        self.n_dags_ = self.dags_.shape[0]
        self.Xs_ = []

    def add_environment(self, X):
        X = np.asarray(X)
        if self.n_envs_ == 0:
            self.pvalues_ = np.ones((self.n_dags_, self.n_vars_, 1, 1))
        else:
            old_changes = self.pvalues_.copy()
            self.pvalues_ = np.ones((self.n_dags_, self.n_vars_, self.n_envs_+1, self.n_envs_+1))
            self.pvalues_[:, :, :self.n_envs_, :self.n_envs_] = old_changes
        
        for env, prior_X in enumerate(self.Xs_):
            try:
                pvalues = test_dag_shifts(  # shape (n_dags, n_mech, 2, 2)
                    Xs=[prior_X, X],
                    dags=self.dags_,
                    test=self.test,
                    test_kwargs=self.test_kwargs)
                self.pvalues_[:, :, -1, env] = pvalues[:, :, 0, 1]
                self.pvalues_[:, :, env, -1] = pvalues[:, :, 0, 1]
            except ValueError as e:
                print(e)
                self.pvalues_[:, :, -1, env] = 1
                self.pvalues_[:, :, env, -1] = 1


        self.n_envs_ += 1
        self.Xs_.append(X)

    @property
    def n_dag_changes_(self):
        return np.sum(self.pvalues_ <= self.alpha_, axis=(1, 2, 3)) / 2

    @property
    def soft_scores_(self):
        scores = np.sum(1 - self.pvalues_, axis=(1, 2, 3))

        return scores

    def get_min_dags(self, soft=False):
        if soft:
            scores = self.soft_scores_
            min_idx = np.where(scores == np.min(scores))[0]
        else:
            min_idx = np.where(self.n_dag_changes_ == np.min(self.n_dag_changes_))[0]
        return self.dags_[min_idx]

    def get_min_cpdag(self, soft=False):
        min_dags = self.get_min_dags(soft=soft)
        cpdag = (np.sum(min_dags, axis=0) > 0).astype(int)
        return cpdag


def _construct_augmented_cpdag(cpdag):
    from sparse_shift.utils import create_causal_learn_cpdag
    from sparse_shift.causal_learn.GraphClass import CausalGraph
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    n_x_vars = cpdag.shape[0]
    cl_cpdag = create_causal_learn_cpdag(cpdag)
    cl_cpdag.add_node(GraphNode(f'X{n_x_vars+1}'))

    nodes = cl_cpdag.get_nodes()

    for i in range(n_x_vars):
        cl_cpdag.add_edge(Edge(nodes[i], nodes[-1], Endpoint.ARROW, Endpoint.TAIL))

    cg = CausalGraph(G=cl_cpdag)

    return cg


class AugmentedPC:
    """
    Runs the PC algorithm on an augmented graph, starting from a known MEC (optional).
    """
    def __init__(self, cpdag, test='kci', alpha=0.05, test_kwargs={}, verbose=False):
        self.cpdag = cpdag
        self.test = test
        self.alpha = alpha
        self.test_kwargs = test_kwargs
        self.dags_ = np.asarray(cpdag2dags(cpdag))
        self.n_vars_ = cpdag.shape[0]
        self.alpha_ = alpha
        self.n_envs_ = 0
        self.n_dags_ = self.dags_.shape[0]
        self.aug_cpdag_ = _construct_augmented_cpdag(cpdag)
        self.Xs_ = []
        self.verbose = verbose

    def add_environment(self, X):
        self.Xs_.append(np.asarray(X))

        if len(self.Xs_) == 1:
            self.learned_cpdag_ = self.cpdag
            return

        data = np.block([
            [self.Xs_[e], np.reshape([e] * self.Xs_[e].shape[0], (-1, 1))]
            for e in range(len(self.Xs_))
        ])

        if self.test == 'fisherz':
            from causallearn.utils.cit import fisherz
            test_func = fisherz
        elif self.test == 'kci':
            from causallearn.utils.cit import kci
            test_func = kci
        else:
            raise ValueError(f'Test {self.test} not implemented.')


        # Run meek rules on found edges
        from sparse_shift.causal_learn.SkeletonDiscovery import augmented_skeleton_discovery
        cg_skel_disc = augmented_skeleton_discovery(data, self.alpha_, test_func,
            stable=True,
            background_knowledge=None, verbose=self.verbose,
            show_progress=self.verbose, cg=self.aug_cpdag_)

        from causallearn.utils.PCUtils import Meek
        cg_meek = Meek.meek(cg_skel_disc, background_knowledge=None)

        adj = np.zeros(cg_meek.G.graph.shape)
        adj[cg_meek.G.graph > 0] = 1
        adj[np.abs(cg_meek.G.graph + cg_meek.G.graph.T) > 0] = 1

        self.learned_cpdag_ = adj[:-1, :-1]

    def get_min_dags(self, soft=None):
        """For experiment compliance"""
        return self.get_dags()

    def get_min_cpdag(self, soft=None):
        """For experiment compliance"""
        return self.get_cpdag()

    def get_dags(self, soft=None):
        return cpdag2dags(self.get_cpdag())

    def get_cpdag(self, soft=None):
        return self.learned_cpdag_
