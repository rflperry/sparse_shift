import numpy as np
import pickle


def get_triangle_dags():
    """Gets all graphs on 3 variables"""
    dags = []
    for e1 in [-1, 0, 1]:  # x1 - x2
        for e2 in [-1, 0, 1]:  # x2 - x3
            for e3 in [-1, 0, 1]:  # x3 - x1
                # skip if cyclic
                if np.abs(np.sum([e1, e2, e3])) == 3:
                    continue
                # add edges
                dag = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                for i, e in enumerate([e1, e2, e3]):
                    if e == 1:  # X -> Y
                        dag[i, (i + 1) % 3] = 1
                    elif e == -1:  # X <- Y
                        dag[(i + 1) % 3, i] = 1
                dags.append(dag)
    return dags


dags = get_triangle_dags()

dag_dict = {f'DAG-{i}': dag for i, dag in enumerate(dags)}

with open("./dag_dict_all_triangles.pkl", "wb") as f:
    pickle.dump(dag_dict, f)
