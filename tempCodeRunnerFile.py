logging.info("generating augmented graph")
    # graph_aug = augmentation(graph)
    # logging.info("partition augmented graph")
    # communities_aug = [graph_aug.subgraph(comm_nodes) for comm_nodes in partition_graph(graph_aug)]
    # logging.info("get {} augmented communities".format(len(communities_aug)))
    # logging.info("generating augmented refrences")
    # references_aug = generate_null_model(num_models=num_references, min_size=5, n=n, p=p, augment=True)
    # logging.info("generating augmented null samples")
    # null_samples_aug = generate_null_model(num_models=num_samples, min_size=5, n=n, p=p, augment=True)
    # matrix_names = ['upper', 'lower', 'comb', 'rw']
    # for comm_idx, community in enumerate(communities_aug):
    #     logging.info("computing matrix scores for community No.{}".format(comm_idx))
    #     matrix_scores = compute_matrix_score(community, references_aug, null_samples_aug) # 4 tuples of (score1, score2)
    #     for matrix_idx, matrix_name in enumerate(matrix_names):
    #         for node in community.nodes():
    #             graph.node[node]['{}_1'.format(matrix_name)] = matrix_scores[matrix_idx][0][node]
    #             graph.node[node][