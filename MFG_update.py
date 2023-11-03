#Itunuoluwa Isewon, Stephen Binaansim, Faith Adegoke, Jerry Emmanuel and Jelili Oyelade1
#"Machine learning methods for predicting essential metabolic genes from Plasmodium falciparum genome-scale metabolic network"  

#Lilli J Freischem, Mauricio Barahona, Diego A Oyarzún
# "Prediction of gene essentiality using machine learning and genome-scale metabolic models"
# Proceedings of Foundations of Systems Biology and Engineering (FOSBE), Bostom, 2022.

# Mariando Beguerisse-Díaz et al
# "Flux-dependent graphs for metabolic networks"
# NPJ Syst Biol Appl 4, 32 (2018).


class MFG_update:
    """Class representation for a mass flow graph
    Parameters
    ----------
    model : cobra.Model
        An existing cobra Model object corresponding to the MFG instance.
    solution : cobra.Solution, optional
        A cobra Solution for building the MFG.
        [if not provided, it will be computed from the model with FBA]

    Attributes
    ----------
    model : cobra.Model
        The cobra model corresponding to the MFG instance.
    nodes : pandas.DataFrame
        A DataFrame of nodes of the Wild Type (WT) graph with columns: id, label
        [and optional columns: pagerank, betweenness, essentiality, outliers].
    edges : pandas.DataFrame
        A DataFrame of the edges of the WT graph with columns: source, target, weight.
    matrix : numpy.array
        A numpy array of the adjacency matrix of the WT graph.
    solution : cobra.Solution
        The solution obtained from optimizing the WT model.
    v : numpy.array
        A numpy array of the flux vector of the solution.
    S2m_plus : numpy.array
        A numpy array of the stoichiometrix production matrix.
    S2m_minus : numpy.array
        A numpy array of the stoichiometrix consumption matrix.
    m : int
        An integer stating the number of reactions in the model.
    """
    
    
    def __init__(self, model, solution = None):

        if not isinstance(model, cobra.Model):
            raise TypeError(f'Input model must be a valid cobra model, '
                            f'not {type(model)}')

        self.model = model

        # get stoichiometric matrix from model
        S = array.create_stoichiometric_matrix(model)
        m = S.shape[1]
        self.m = m

        # reversibility vector
        rlist = [r.reversibility for r in model.reactions]
        r = np.zeros([m, 1])
        r[rlist] = 1
        R = np.diag(r[:,0])

        # production and consumption matrices
        Im = np.identity(m)
        S2m_1 = np.block([S, -S])
        S2m_2 = np.block([[Im, np.zeros([m,m])], [np.zeros([m,m]), R]])
        S2m = S2m_1@S2m_2
        abs_S2m = np.abs(S2m)
        self.S2m_plus = 0.5 * (abs_S2m + S2m)
        self.S2m_minus = 0.5 * (abs_S2m - S2m)

        # production and consumption fluxes
        self.solution = solution

        # compute solution if not provided
        if self.solution is None:
            # * changed to FBA *
            self.solution = self.model.optimize()

        self.v = self.solution.fluxes.values.reshape([m,1])
        self.v2m = self.compute_v2m(self.v)

        self.matrix = self.compute_mfg(self.v)

        self.nodes, self.edges = self.compute_nodes_and_edges(self.matrix)
        self.nodes = self.centralities(self.nodes, self.edges)

    def compute_v2m(self, v):
        abs_v = np.abs(v)
        v_plus = 0.5 * (abs_v + v).T
        v_minus = 0.5 * (abs_v - v).T
        v2m = np.block([v_plus, v_minus]).T
        return v2m

    def compute_mfg(self, v):
        # production and consumption fluxes
        v2m = self.compute_v2m(v)
        V = np.diag(v2m[:,0]) #[:,0]

        j_v = self.S2m_plus@v2m
        J_v = np.diag(j_v[:,0]) #[:,0]

        # MFG adjacency matrix
        inverse_J = pinv(J_v)
        mfg = (self.S2m_plus@V).T@inverse_J@(self.S2m_minus@V)
        return mfg

    def compute_nodes_and_edges(self, matrix):
        # MFG nodes and edges
        # reaction ids
        ids = np.arange(0,2*self.m).reshape([2*self.m,1])

        # reaction labels
        labels = np.array([self.model.reactions[i].id for i in range(self.m)])
        labels = np.hstack([labels,labels]).reshape([2*self.m,1])

        # edges
        edgesarray = np.vstack((np.where(matrix>0), matrix[np.where(matrix>0)])).T
        edges = pd.DataFrame(data = edgesarray).rename(columns={0:'Source', 1:'Target', 2:'Weight'})
        edges = edges.astype({'Source' : 'int', 'Target' : 'int'})

        # get only active reactions
        active = np.unique(edgesarray[:,0:2].reshape(-1).astype('int'))
        nodes = np.block([ids[active], labels[active]])
        nodes = pd.DataFrame(data=nodes).rename(columns={0:'id', 1:'label'})
        return nodes, edges


    def centralities(self, nodes, edges):
        """
        Computes pagerank and betweenness centrality of nodes.

        Parameters
        ----------
        nodes : pandas.DataFrame
            A DataFrame of nodes of the Wild Type (WT) graph with columns: id, label
        edges : pandas.DataFrame
            A DataFrame of the edges of the WT graph with columns: source, target, weight.

        Returns
        -------
        nodes :
            A DataFrame of nodes of the Wild Type (WT) graph with columns:
                id, label, pagerank, pagerank percentile, betweenness

        """

        #compute pagerank
        G = nx.from_pandas_edgelist(edges, 'Source', 'Target')
        
        pr = nx.pagerank(G, weight='weight')
        prs = [pr[1] for pr in sorted(pr.items())]
        nodes['pagerank']=prs

        # compute pagerank percentiles
        pr_sorted = sorted(prs)
        perc = nodes['pagerank'].apply(lambda x: percentileofscore(pr_sorted, x))
        nodes['pagerank percentile'] = perc

        # compute betweenness centrality
        bt = nx.betweenness_centrality(G, weight='weight')
        bts = [bt[1] for bt in sorted(bt.items())]
        nodes['Betweenness']=bts
        
        # closeness centrality
        cc = nx.closeness_centrality(G)
        ccs = [cc[1] for cc in sorted(cc.items())]
        nodes['closeness']=ccs
        
        # Compute the degree and clustering coefficient for each node in the graph
        dg = dict(G.degree())
        dgs = [dg[1] for dg in sorted(dg.items())]
        nodes['Degree'] = dgs
        
        cl_coeff = nx.clustering(G, weight='weight')
        ccff = [cl_coeff[1] for cl_coeff in sorted(cl_coeff.items())]
        nodes['Clustering_Coefficient'] = ccff
        
        #Edge Clustering Coefficient Centrality
        # edge_cl= nx.edge_clustering(G, weight='weight')
        # edg_cl = [edge_cl[1] for edge_cl in sorted(edge_cl.items())]
        # nodes['Edge_CoC'] = edg_cl


        # Load Centrality
        load_centrality = nx.load_centrality(G, weight='weight')
        ld_C = [load_centrality[1]  for load_centrality in sorted(load_centrality.items())]
        nodes['Load_Centrality'] = ld_C
        
        # Random Walk Betweenness Centrality
        random_wbt = nx.current_flow_betweenness_centrality(G, weight='weight')
        Rn_wbt = [random_wbt[1] for random_wbt in sorted(random_wbt.items())]
        nodes['Random_Walk Betweenness'] = Rn_wbt
        
        # Information Centrality
        information_centrality = nx.information_centrality(G, weight='weight')
        if_cent = [information_centrality[1] for information_centrality in sorted(information_centrality.items())]
        nodes['Information_Centrality'] = if_cent
        
        #Harmonic Centrality
        harmonic = nx.harmonic_centrality(G)
        ham_ct = [harmonic[1] for harmonic in sorted(harmonic.items())]
        nodes['Harmonic_Centrality'] = ham_ct
        
        # Subgraph Centrality
        subgraph = nx.subgraph_centrality(G)
        subg = [subgraph[1] for subgraph in sorted(subgraph.items())]
        nodes['Subgraph_Centrality'] = subg

        # Eigenvector Centrality
        eigenvector = nx.eigenvector_centrality(G, weight='weight')
        eig_c = [eigenvector[1] for eigenvector in sorted(eigenvector.items())]
        nodes['Eigenvector_Cebtrality'] = eig_c
                
        return nodes


    def compute_outliers(self, solution, pr_wt, threshold = 10):
        """
        Computes outliers of all nodes in the network.

        Parameters
        ----------
        solution : cobra.solution
            The solution for the knockout mfg.
        pr_wt : list
            list of pagerank percentiles of each node in the wild type mfg.
        threshold : int, optional
            The pagerank percentile threshold for when a reaction is considered
            to be an outlier. The default is 10.

        Returns
        -------
        outliers : dict
            dictionary containing positive and negative outliers.

        """

        v_ko = solution.fluxes.values.reshape([self.m,1])

        mfg_ko = self.compute_mfg(v_ko)
        nodes_ko, edges_ko = self.compute_nodes_and_edges(mfg_ko)
        nodes_ko = self.centralities(nodes_ko, edges_ko)

        # extract pagerank percentiles
        pr_ko = dict(zip(nodes_ko['id'], nodes_ko['pagerank percentile']))

        outliers = {'pos': [], 'neg': []}
        for k in pr_wt.keys():
            if k in pr_ko.keys():
                pr_diff = pr_wt[k] - pr_ko[k]

                if pr_diff > threshold: # pr_wt at least t larger than pr_ko -> neg change
                    outliers['neg'].append((k, pr_diff))

                if pr_diff < -1 * threshold:
                    outliers['pos'].append((k, pr_diff))

        return outliers

    def analyze(self, outliers=True):
        """
        Analyze essentiality and outliers of all nodes in the MFG.
        Parameters
        ----------
        outliers : bool
            A boolean indicating whether outliers should be computed, default True.

        Returns
        -------
        None.

        """
        print('Starting outlier and essentiality analysis!')
        r_progress = 0

        if outliers:
            if 'pagerank' not in self.nodes.columns:
                self.nodes = self.centralities()
            pr_wt = dict(zip(self.nodes['id'], self.nodes['pagerank percentile']))

        f_WT = self.solution.objective_value # flux through biomass
        essentiality = []
        outliers_pos = []
        outliers_neg = []

        for reaction in self.nodes.label:

            # display progress
            print(f'Analysing reaction # {r_progress} : {reaction}')
            r_progress += 1


            with self.model as model:
                r = model.reactions.get_by_id(reaction)
                r.knock_out()
                # *changed from pFBA to FBA because slim_optimise is a lot faster*
                try:
                    if not outliers:
                        f_KO = model.slim_optimize()
                    if outliers:
                        solution = model.optimize()
                        f_KO = solution.objective_value
                        outliers_KO = self.compute_outliers(solution, pr_wt)
                    # if no solution exists, f_KO is nan - in that case set it to 0
                    if np.isnan(f_KO):
                        f_KO = 0

                except Exception:
                    print('no feasible solution after knockout of ' + reaction)
                    f_KO = 0
                    if outliers:
                        outliers_KO = ({'pos': [], 'neg': []})

                finally:
                    essentiality.append(1-(f_KO/f_WT))

                    if outliers:
                        outliers_pos.append(outliers_KO['pos'])
                        outliers_neg.append(outliers_KO['neg'])

        self.nodes['essentiality'] = essentiality
        if outliers:
            self.nodes['outliers (pos)'] = outliers_pos
            self.nodes['outliers (neg)'] = outliers_neg

    @property
    def outliers(self):
        if 'outliers' not in self.nodes.columns:
            raise Exception('Outliers have not been computed yet. Call MFG.analyze() to compute the outliers of each node.')

        outliers = self.nodes['outliers']
        return outliers

    @property
    def essentiality(self):
        if 'essentiality' not in self.nodes.columns:
            raise Exception('Essentiality has not been computed yet. Call MFG.analyze() to compute the essentiality of each node.')

        essentiality = self.nodes['essentiality']
        return essentiality

    def draw(self):
        plt.figure(figsize=(5, 5))
        G = nx.DiGraph()

        for s,t,w in zip(self.edges['Source'], self.edges['Target'], self.edges['Weight']):
            G.add_weighted_edges_from([(s,t, w)])

        pos = nx.spring_layout(G)

        nx.draw(G, pos = pos, node_size=20, with_labels=False, width=0.3)
        plt.savefig("mygraph.png")


    def cluster(self, draw = False):
        """
        Cluster the MFG using the markov clustering algorithm.

        Parameters
        ----------
        draw : bool
            A boolean indicating whether or not to visualize the clustering, default False.

        Returns
        -------
        clusters : List

        """
        G = nx.DiGraph()

        for s,t,w in zip(self.edges['Source'], self.edges['Target'], self.edges['Weight']):
            G.add_weighted_edges_from([(s,t, w)])

        matrix = nx.to_scipy_sparse_matrix(G)
        result = mc.run_mcl(matrix)           # run MCL with default parameters
        clusters = mc.get_clusters(result)    # get clusters

        if draw:
            plt.figure()
            # translate positions
            pos = nx.spring_layout(G)
            tr = list(pos.keys())

            pos = [pos[tr[i]] for i in range(len(tr))]
            mc.draw_graph(matrix, clusters, pos = pos, node_size=20, with_labels=False,
                        cmap = cm.OrRd_r, width=0.3)
        return clusters

    def export(self, filename='mfg', directory='', matrix=True, nodes=True, edges=True):
        """
        Function to export the MFG.

        Parameters
        ----------
        filename : string, optional
            the file name for saving the mfg. The default is 'mfg'.
        directory : string, optional
            the destination directory for the mfg files. The default is ''.
        matrix : Boolean, optional
            Whether or not to save the adjacency matrix. The default is True.
        nodes : Boolean, optional
            Whether or not to save the nodes table. The default is True.
        edges : TYPE, Boolean, optional
            Whether or not to save the edges table. The default is True.

        Returns
        -------
        None.

        """
        if matrix: np.save(f'{directory}{filename}.npy', sparse.csr_matrix(self.matrix))
        if nodes: self.nodes.to_csv(f'{directory}{filename}_nodes.csv')
        if edges: self.edges.to_csv(f'{directory}{filename}_edges.csv')


